"""
Evaluation utilities for JointQuant
Adapted from DartQuant/fake_quant/eval_utils.py
"""

import torch
import torch.nn as nn
import math
import logging
from tqdm import tqdm
from typing import Optional

from . import utils
from . import model_utils
from . import data_utils


@torch.no_grad()
def evaluate_perplexity(
    model: nn.Module,
    tokenizer,
    device: torch.device,
    dataset: str = 'wikitext2',
    seqlen: int = 2048,
    batch_size: int = 1
) -> float:
    """
    Evaluate perplexity on a dataset.
    
    Args:
        model: The model to evaluate
        tokenizer: Tokenizer
        device: Device to run on
        dataset: Dataset name ('wikitext2', 'c4', 'ptb')
        seqlen: Sequence length
        batch_size: Batch size for evaluation
    
    Returns:
        Perplexity value
    """
    logging.info(f"Evaluating perplexity on {dataset}...")
    
    model.eval()
    model.seqlen = seqlen
    
    # Load test data
    testenc = data_utils.get_loaders(dataset, tokenizer=tokenizer, eval_mode=True)
    
    # Get input IDs
    input_ids = testenc.input_ids  # [1, text_len]
    nsamples = input_ids.numel() // seqlen
    
    if nsamples == 0:
        logging.warning("Not enough data for evaluation")
        return float('inf')
    
    # Truncate to exact number of samples
    input_ids = input_ids[:, :nsamples * seqlen].view(nsamples, seqlen)
    
    # Create batches
    batches = [input_ids[i:i+batch_size] for i in range(0, nsamples, batch_size)]
    
    nlls = []
    loss_fct = nn.CrossEntropyLoss(reduction="none")
    
    for batch in tqdm(batches, desc="Evaluating"):
        batch = batch.to(device)
        
        with torch.no_grad():
            outputs = model(batch)
            logits = outputs.logits if hasattr(outputs, 'logits') else outputs[0]
        
        # Check for invalid values
        if torch.isnan(logits).any() or torch.isinf(logits).any():
            logging.warning("NaN/Inf in logits, skipping batch")
            continue
        
        # Shift for next token prediction
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = batch[:, 1:].contiguous()
        
        # Calculate cross entropy per token
        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1)
        )
        
        # Mean per sequence
        loss = loss.view(batch.size(0), -1).mean(dim=1)
        
        # Skip invalid losses
        valid_mask = ~(torch.isnan(loss) | torch.isinf(loss) | (loss > 100))
        if valid_mask.any():
            nlls.append(loss[valid_mask])
    
    if len(nlls) == 0:
        logging.error("No valid batches evaluated")
        return float('inf')
    
    # Calculate perplexity
    nlls_tensor = torch.cat(nlls)
    avg_nll = nlls_tensor.mean().item()
    
    # Clamp to prevent overflow
    avg_nll = min(avg_nll, 20.0)
    ppl = math.exp(avg_nll)
    
    logging.info(f'{dataset.upper()} PPL: {ppl:.2f}')
    return ppl


@torch.no_grad()
def evaluate_perplexity_simple(
    model: nn.Module,
    tokenizer,
    seqlen: int = 2048,
    device: torch.device = None,
    max_samples: int = 64,
    dataset: str = 'wikitext2'
) -> float:
    """
    Simple perplexity evaluation - loads data automatically.
    
    Args:
        model: The model to evaluate
        tokenizer: Tokenizer
        seqlen: Sequence length for evaluation
        device: Device to run on (auto-detect if None)
        max_samples: Maximum number of samples to evaluate
        dataset: Dataset name ('wikitext2', 'c4', 'ptb')
    
    Returns:
        Perplexity value
    """
    model.eval()
    
    # Auto-detect device
    if device is None:
        device = next(model.parameters()).device
    
    # Load test data
    testenc = data_utils.get_loaders(dataset, tokenizer=tokenizer, eval_mode=True)
    test_data = testenc.input_ids  # [1, text_len]
    
    # Reshape if needed
    if test_data.dim() == 1:
        test_data = test_data.unsqueeze(0)
    
    if test_data.shape[0] == 1:
        # Single long sequence - split into chunks
        total_tokens = test_data.shape[1]
        nsamples = total_tokens // seqlen
        if nsamples == 0:
            nsamples = 1
            seqlen = total_tokens
        test_data = test_data[:, :nsamples * seqlen].view(nsamples, seqlen)
    
    if max_samples is not None:
        test_data = test_data[:max_samples]
    
    nlls = []
    
    for i in tqdm(range(len(test_data)), desc="Evaluating"):
        batch = test_data[i:i+1].to(device)
        
        with torch.no_grad():
            outputs = model(batch)
            logits = outputs.logits if hasattr(outputs, 'logits') else outputs[0]
        
        if torch.isnan(logits).any() or torch.isinf(logits).any():
            continue
        
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = batch[:, 1:].contiguous()
        
        loss = nn.functional.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            reduction='mean',
            ignore_index=tokenizer.pad_token_id if tokenizer.pad_token_id else -100
        )
        
        if not (torch.isnan(loss) or torch.isinf(loss) or loss.item() > 100):
            nlls.append(loss.item())
    
    if len(nlls) == 0:
        return float('inf')
    
    avg_nll = sum(nlls) / len(nlls)
    avg_nll = min(avg_nll, 20.0)
    ppl = math.exp(avg_nll)
    
    return ppl
