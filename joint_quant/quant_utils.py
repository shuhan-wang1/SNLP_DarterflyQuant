"""
Quantization utilities for JointQuant
Adapted from DartQuant/fake_quant/quant_utils.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Optional


def get_minq_maxq(bits: int, sym: bool) -> Tuple[torch.Tensor, torch.Tensor]:
    """Get min and max quantization values for given bit width"""
    if sym:
        maxq = torch.tensor(2**(bits - 1) - 1)  # 7 for 4-bit
        minq = -maxq - 1  # -8 for 4-bit
    else:
        maxq = torch.tensor(2**bits - 1)  # 15 for 4-bit
        minq = torch.tensor(0)
    return minq, maxq


# ============================================================================
# Symmetric Quantization (for weights)
# ============================================================================

def sym_quant(x: torch.Tensor, scale: torch.Tensor, maxq: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Symmetric quantization: q = clamp(round(x / scale), -(maxq+1), maxq)"""
    scale = scale.to(x.device)
    q = torch.clamp(torch.round(x / scale), -(maxq + 1), maxq)  # [-8, 7] for 4-bit
    return q, scale


def sym_dequant(q: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Symmetric dequantization: x = scale * q"""
    return scale * q


def sym_quant_dequant(x: torch.Tensor, scale: torch.Tensor, maxq: torch.Tensor) -> torch.Tensor:
    """Quantize and immediately dequantize (for fake quantization)"""
    q, scale = sym_quant(x, scale, maxq)
    return sym_dequant(q, scale)


# ============================================================================
# Asymmetric Quantization (for activations)
# ============================================================================

def asym_quant(x: torch.Tensor, scale: torch.Tensor, zero: torch.Tensor, 
               maxq: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Asymmetric quantization: q = clamp(round(x / scale) + zero, 0, maxq)"""
    scale = scale.to(x.device)
    zero = zero.to(x.device)
    q = torch.clamp(torch.round(x / scale) + zero, 0, maxq)
    return q, scale, zero


def asym_dequant(q: torch.Tensor, scale: torch.Tensor, zero: torch.Tensor) -> torch.Tensor:
    """Asymmetric dequantization: x = scale * (q - zero)"""
    return scale * (q - zero)


def asym_quant_dequant(x: torch.Tensor, scale: torch.Tensor, zero: torch.Tensor, 
                       maxq: torch.Tensor) -> torch.Tensor:
    """Quantize and immediately dequantize (for fake quantization)"""
    q, scale, zero = asym_quant(x, scale, zero, maxq)
    return asym_dequant(q, scale, zero)


# ============================================================================
# Weight Quantizer (per-channel or group-wise symmetric)
# ============================================================================

class WeightQuantizer(nn.Module):
    """
    Weight quantizer with per-channel or group-wise symmetric quantization.
    Matches official DartQuant/GPTQ implementation.
    """
    
    def __init__(self, shape: int = 1):
        super().__init__()
        self.register_buffer('maxq', torch.tensor(0))
        self.register_buffer('scale', torch.zeros(shape))
        self.register_buffer('zero', torch.zeros(shape))
        self.bits = 16
        self.groupsize = -1
        self.sym = True
        self.mse = False
        
    def configure(self, bits: int, perchannel: bool = True, sym: bool = True,
                  mse: bool = False, groupsize: int = -1):
        """Configure quantization parameters"""
        self.bits = bits
        self.perchannel = perchannel
        self.sym = sym
        self.mse = mse
        self.groupsize = groupsize
        
        if sym:
            self.maxq = torch.tensor(2**(bits - 1) - 1)  # 7 for 4-bit
        else:
            self.maxq = torch.tensor(2**bits - 1)  # 15 for 4-bit
    
    def find_params(self, x: torch.Tensor):
        """Find quantization parameters for input tensor"""
        if self.bits == 16:
            return
        
        dev = x.device
        self.maxq = self.maxq.to(dev)
        
        shape = x.shape
        if self.perchannel:
            x = x.flatten(1)
        else:
            x = x.flatten().unsqueeze(0)
        
        tmp = torch.zeros(x.shape[0], device=dev)
        xmin = torch.minimum(x.min(1)[0], tmp)
        xmax = torch.maximum(x.max(1)[0], tmp)
        
        if self.sym:
            xmax = torch.maximum(torch.abs(xmin), xmax).clamp(min=1e-5)
            self.scale = xmax / self.maxq
            self.zero = torch.zeros_like(self.scale)
        else:
            tmp = (xmin == 0) & (xmax == 0)
            xmin.masked_fill(tmp, -1)
            xmax.masked_fill(tmp, +1)
            self.scale = (xmax - xmin).clamp(min=1e-5) / self.maxq
            self.zero = torch.round(-xmin / self.scale)
        
        if not self.perchannel:
            tmp = shape[0]
            self.scale = self.scale.repeat(tmp)
            self.zero = self.zero.repeat(tmp)
        
        shape = [-1] + [1] * (len(shape) - 1)
        self.scale = self.scale.reshape(shape)
        self.zero = self.zero.reshape(shape)
    
    def quantize(self, x: torch.Tensor) -> torch.Tensor:
        """Quantize and dequantize (fake quantization)"""
        x_dtype = x.dtype
        if self.bits < 16 and torch.all(self.scale != 0):
            if self.sym:
                return sym_quant_dequant(x, self.scale, self.maxq).to(x_dtype)
            return asym_quant_dequant(x, self.scale, self.zero, self.maxq).to(x_dtype)
        return x
    
    def enabled(self) -> bool:
        return self.maxq > 0
    
    def ready(self) -> bool:
        return torch.all(self.scale != 0)


# ============================================================================
# Activation Quantizer (per-token symmetric or asymmetric)
# ============================================================================

class ActQuantizer(nn.Module):
    """
    Activation quantizer with per-token quantization.
    Matches official DartQuant implementation.
    """
    
    def __init__(self):
        super().__init__()
        self.register_buffer('maxq', torch.tensor(0))
        self.register_buffer('scale', torch.zeros(1))
        self.register_buffer('zero', torch.zeros(1))
        self.bits = 16
        self.groupsize = -1
        self.sym = False
        self.clip_ratio = 1.0
    
    def free(self):
        """Free buffers to save memory"""
        self.zero = None
        self.scale = None
    
    def configure(self, bits: int, groupsize: int = -1, sym: bool = False,
                  clip_ratio: float = 1.0):
        """Configure quantization parameters"""
        _, self.maxq = get_minq_maxq(bits, sym)
        self.bits = bits
        self.groupsize = groupsize
        self.sym = sym
        self.clip_ratio = clip_ratio
        assert 0 < clip_ratio <= 1, 'Clip ratio should be in (0, 1]'
    
    def find_params(self, x: torch.Tensor):
        """Find quantization parameters for input tensor"""
        if self.bits == 16:
            return
        
        dev = x.device
        self.maxq = self.maxq.to(dev)
        init_shape = x.shape
        
        if self.groupsize > 0:
            # Group-wise per-token quantization
            self.find_params_per_token_groupwise(x)
            return
        
        # Per-token quantization
        reshaped_x = x.reshape((-1, x.shape[-1]))
        
        tmp = torch.zeros(reshaped_x.shape[0], device=dev)
        xmin = torch.minimum(reshaped_x.min(1)[0], tmp) * self.clip_ratio
        xmax = torch.maximum(reshaped_x.max(1)[0], tmp) * self.clip_ratio
        
        if self.sym:
            xmax = torch.maximum(torch.abs(xmin), xmax)
            tmp = xmax == 0
            self.scale = (xmax / self.maxq).unsqueeze(1).repeat(1, reshaped_x.shape[-1])
            self.scale[tmp] = 1
            self.scale = self.scale.reshape(init_shape)
            self.zero = torch.zeros_like(self.scale)
        else:
            tmp = (xmin == 0) & (xmax == 0)
            xmin[tmp] = -1
            xmax[tmp] = +1
            self.scale = (xmax - xmin) / self.maxq
            self.zero = torch.round(-xmin / self.scale)
            
            self.scale = self.scale.unsqueeze(1).repeat(1, reshaped_x.shape[-1]).reshape(init_shape)
            self.zero = self.zero.unsqueeze(1).repeat(1, reshaped_x.shape[-1]).reshape(init_shape)
    
    def find_params_per_token_groupwise(self, x: torch.Tensor):
        """Find parameters for group-wise per-token quantization"""
        init_shape = x.shape
        reshaped_x = x.reshape(-1, x.shape[-2], x.shape[-1] // self.groupsize, self.groupsize)
        
        xmax = torch.amax(reshaped_x, dim=3, keepdim=True) * self.clip_ratio
        xmin = torch.amin(reshaped_x, dim=3, keepdim=True) * self.clip_ratio
        
        if self.sym:
            xmax = torch.maximum(torch.abs(xmin), xmax)
            tmp = xmax == 0
            self.scale = xmax / self.maxq
            self.scale[tmp] = 1
            self.zero = torch.zeros_like(self.scale)
        else:
            tmp = (xmin == 0) & (xmax == 0)
            xmin[tmp] = -1
            xmax[tmp] = +1
            self.scale = (xmax - xmin) / self.maxq
            self.zero = torch.round(-xmin / self.scale)
        
        self.scale = self.scale.repeat(1, 1, 1, self.groupsize).reshape(init_shape)
        self.zero = self.zero.repeat(1, 1, 1, self.groupsize).reshape(init_shape)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: find params and quantize"""
        x_dtype = x.dtype
        
        if self.bits == 16:
            return x
        
        self.find_params(x)
        
        if self.sym:
            return sym_quant_dequant(x, self.scale, self.maxq).to(x_dtype)
        else:
            return asym_quant_dequant(x, self.scale, self.zero, self.maxq).to(x_dtype)


# ============================================================================
# Quantized Linear Layer (Group-wise W4A4)
# ============================================================================

class QuantizedLinear(nn.Module):
    """
    Quantized linear layer with INT4 weights (Group-wise) and Asymmetric Activations.
    
    Key features matching official DartQuant:
    1. Group-wise weight quantization (group_size=128)
    2. Symmetric weight quantization with full [-8, 7] range
    3. Per-token asymmetric activation quantization
    """
    
    def __init__(self, weight: torch.Tensor, bias: Optional[torch.Tensor] = None,
                 w_bits: int = 4, a_bits: int = 4, group_size: int = 128):
        super().__init__()
        self.w_bits = w_bits
        self.a_bits = a_bits
        self.group_size = group_size
        self.has_bias = bias is not None
        
        out_features, in_features = weight.shape
        self.out_features = out_features
        self.in_features = in_features
        
        # Pad weight if input features not divisible by group size
        if in_features % group_size != 0:
            pad_len = group_size - (in_features % group_size)
            weight = F.pad(weight, (0, pad_len))
            self.padded_in_features = in_features + pad_len
        else:
            self.padded_in_features = in_features
        
        # Reshape to [out_features, num_groups, group_size]
        num_groups = self.padded_in_features // group_size
        weight_reshaped = weight.float().reshape(out_features, num_groups, group_size)
        
        # Calculate scales per group (Symmetric quantization)
        # For 4-bit symmetric: range is [-8, 7], so use 8.0 as divisor
        # This matches official DartQuant: maxq = 7, clamp to -(maxq+1), maxq
        max_val = weight_reshaped.abs().amax(dim=-1, keepdim=True)
        scale = torch.clamp(max_val / 8.0, min=1e-8)
        
        # Quantize with full INT4 range [-8, 7]
        weight_q = torch.clamp(torch.round(weight_reshaped / scale), -8, 7)
        
        # Store quantized weights and scales
        self.register_buffer('quantized_weight', weight_q.to(torch.int8))
        self.register_buffer('weight_scale', scale.squeeze(-1).to(weight.dtype))
        
        if self.has_bias:
            self.register_buffer('bias', bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_dtype = x.dtype
        x = x.float()
        
        # Activation Quantization (Asymmetric per-token, safer for A4)
        if self.a_bits <= 8:
            x_min = x.min(dim=-1, keepdim=True).values
            x_max = x.max(dim=-1, keepdim=True).values
            # Asymmetric: 0-15 range for 4-bit unsigned
            scale = torch.clamp((x_max - x_min) / 15.0, min=1e-8)
            zp = torch.round(-x_min / scale)
            x_q = torch.clamp(torch.round(x / scale + zp), 0, 15)
            x = (x_q - zp) * scale
        
        # Weight Dequantization
        weight = self.quantized_weight.float() * self.weight_scale.unsqueeze(-1)
        weight = weight.reshape(self.out_features, self.padded_in_features)
        weight = weight[:, :self.in_features]  # Remove padding
        
        # Linear operation
        output = F.linear(x, weight, self.bias if self.has_bias else None)
        
        return output.to(original_dtype)


# ============================================================================
# Activation Quantization Wrapper
# ============================================================================

class ActQuantWrapper(nn.Module):
    """
    Wrapper for linear layers with activation quantization.
    Matches official DartQuant ActQuantWrapper.
    """
    
    def __init__(self, module: nn.Linear):
        super().__init__()
        assert isinstance(module, nn.Linear)
        self.module = module
        self.weight = module.weight
        self.bias = module.bias
        self.quantizer = ActQuantizer()
        self.out_quantizer = ActQuantizer()
        
        # Hadamard transform settings
        self.register_buffer('had_K', torch.tensor(0))
        self._buffers['had_K'] = None
        self.K = 1
        self.online_full_had = False
        self.online_partial_had = False
        self.had_dim = 0
        self.fp32_had = False
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_dtype = x.dtype
        
        # Input quantization
        if self.quantizer.bits < 16:
            x = self.quantizer(x).to(x_dtype)
            self.quantizer.free()
        
        # Linear
        x = self.module(x).to(x_dtype)
        
        # Output quantization
        if self.out_quantizer.bits < 16:
            x = self.out_quantizer(x).to(x_dtype)
            self.out_quantizer.free()
        
        return x


def add_actquant(module: nn.Module, name: str = '', 
                 layers=[nn.Linear]) -> None:
    """Add ActQuantWrapper to all linear layers in module"""
    if isinstance(module, ActQuantWrapper):
        return
    
    for attr in dir(module):
        tmp = getattr(module, attr)
        if type(tmp) in layers:
            setattr(module, attr, ActQuantWrapper(tmp))
    
    for name1, child in module.named_children():
        add_actquant(child, name + '.' + name1 if name != '' else name1, layers)


def find_qlayers(module: nn.Module, layers=[nn.Linear, ActQuantWrapper], 
                 name: str = '') -> dict:
    """Find all quantizable layers in module"""
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_qlayers(
            child, layers=layers, 
            name=name + '.' + name1 if name != '' else name1
        ))
    return res
