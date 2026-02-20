"""
Utility functions for JointQuant
Adapted from DartQuant/fake_quant/utils.py
"""

import torch
import random
import numpy as np
import os
import gc
import logging
from pathlib import Path

# Disable TensorFloat-32 to avoid numerical issues (from official DartQuant)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

DEV = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')


def set_seed(seed: int):
    """Set random seed for reproducibility"""
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    random.seed(seed)


def cleanup_memory(verbose: bool = True) -> None:
    """Run GC and clear GPU memory."""
    import inspect
    caller_name = ''
    try:
        caller_name = f' (from {inspect.stack()[1].function})'
    except (ValueError, KeyError):
        pass

    def total_reserved_mem() -> int:
        return sum(torch.cuda.memory_reserved(device=i) for i in range(torch.cuda.device_count()))

    memory_before = total_reserved_mem()

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        memory_after = total_reserved_mem()
        if verbose:
            logging.info(
                f"GPU memory{caller_name}: {memory_before / (1024 ** 3):.2f} -> {memory_after / (1024 ** 3):.2f} GB"
                f" ({(memory_after - memory_before) / (1024 ** 3):.2f} GB)"
            )


def llama_down_proj_groupsize(model, groupsize: int) -> int:
    """Calculate groupsize for down_proj layer in Llama models"""
    assert groupsize > 1, 'groupsize should be greater than 1!'

    if model.config.intermediate_size % groupsize == 0:
        logging.info(f'(Act.) Groupsize = Down_proj Groupsize: {groupsize}')
        return groupsize

    group_num = int(model.config.hidden_size / groupsize)
    assert groupsize * group_num == model.config.hidden_size, 'Invalid groupsize for llama!'

    down_proj_groupsize = model.config.intermediate_size // group_num
    assert down_proj_groupsize * group_num == model.config.intermediate_size, 'Invalid groupsize for down_proj!'
    logging.info(f'(Act.) Groupsize: {groupsize}, Down_proj Groupsize: {down_proj_groupsize}')
    return down_proj_groupsize


def config_logging(log_file: str, to_console: bool = True):
    """Configure logging to file and console"""
    log_dir = os.path.dirname(log_file)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)

    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    file_handler = logging.FileHandler(log_file, mode='a')
    file_handler.setFormatter(formatter)

    handlers = [file_handler]
    if to_console:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        handlers.append(console_handler)

    logging.basicConfig(level=logging.INFO, handlers=handlers)
