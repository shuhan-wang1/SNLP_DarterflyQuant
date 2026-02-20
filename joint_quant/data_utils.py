"""
Data utilities for JointQuant
Adapted from DartQuant/fake_quant/data_utils.py
"""

import torch
import random
import logging
from typing import List, Tuple, Optional
from datasets import load_dataset
import transformers


def get_wikitext2(nsamples: int, seed: int, seqlen: int, 
                  tokenizer, eval_mode: bool = False):
    """
    Load WikiText2 dataset.
    
    Args:
        nsamples: Number of calibration samples (ignored in eval_mode)
        seed: Random seed for sample selection
        seqlen: Sequence length
        tokenizer: Tokenizer to use
        eval_mode: If True, return full test set; if False, return random train samples
    
    Returns:
        In eval_mode: tokenized test data
        Otherwise: list of (input_ids, targets) tuples
    """
    if eval_mode:
        testdata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
        testenc = tokenizer("\n\n".join(testdata['text']), return_tensors='pt')
        return testenc
    else:
        traindata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
        trainenc = tokenizer("\n\n".join(traindata['text']), return_tensors='pt')
        
        random.seed(seed)
        trainloader = []
        for _ in range(nsamples):
            i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
            j = i + seqlen
            inp = trainenc.input_ids[:, i:j]
            tar = inp.clone()
            tar[:, :-1] = -100
            trainloader.append((inp, tar))
        return trainloader


def get_c4(nsamples: int, seed: int, seqlen: int,
           tokenizer, eval_mode: bool = False):
    """Load C4 dataset"""
    if eval_mode:
        valdata = load_dataset('allenai/c4', 'en', split='validation', 
                               streaming=True, trust_remote_code=True)
        # Take first 1100 samples
        texts = []
        for i, item in enumerate(valdata):
            if i >= 1100:
                break
            texts.append(item['text'])
        valenc = tokenizer(' '.join(texts), return_tensors='pt')
        valenc = valenc.input_ids[:, :(256 * seqlen)]
        
        class TokenizerWrapper:
            def __init__(self, input_ids):
                self.input_ids = input_ids
        return TokenizerWrapper(valenc)
    else:
        traindata = load_dataset('allenai/c4', 'en', split='train',
                                 streaming=True, trust_remote_code=True)
        
        random.seed(seed)
        trainloader = []
        data_iter = iter(traindata)
        
        for _ in range(nsamples):
            while True:
                try:
                    item = next(data_iter)
                    trainenc = tokenizer(item['text'], return_tensors='pt')
                    if trainenc.input_ids.shape[1] >= seqlen:
                        break
                except StopIteration:
                    data_iter = iter(traindata)
            
            i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
            j = i + seqlen
            inp = trainenc.input_ids[:, i:j]
            tar = inp.clone()
            tar[:, :-1] = -100
            trainloader.append((inp, tar))
        return trainloader


def get_ptb(nsamples: int, seed: int, seqlen: int,
            tokenizer, eval_mode: bool = False):
    """Load Penn Treebank dataset"""
    if eval_mode:
        testdata = load_dataset('ptb_text_only', 'penn_treebank', split='test')
        testenc = tokenizer(" ".join(testdata['sentence']), return_tensors='pt')
        return testenc
    else:
        traindata = load_dataset('ptb_text_only', 'penn_treebank', split='train')
        trainenc = tokenizer(" ".join(traindata['sentence']), return_tensors='pt')
        
        random.seed(seed)
        trainloader = []
        for _ in range(nsamples):
            i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
            j = i + seqlen
            inp = trainenc.input_ids[:, i:j]
            tar = inp.clone()
            tar[:, :-1] = -100
            trainloader.append((inp, tar))
        return trainloader


def get_loaders(name: str, nsamples: int = 128, seed: int = 0, seqlen: int = 2048,
                tokenizer=None, eval_mode: bool = False):
    """
    Get data loaders for specified dataset.
    
    Args:
        name: Dataset name ('wikitext2', 'c4', 'ptb')
        nsamples: Number of calibration samples
        seed: Random seed
        seqlen: Sequence length
        tokenizer: Tokenizer to use
        eval_mode: If True, return evaluation data
    
    Returns:
        Data loader or tokenized data
    """
    if 'wikitext2' in name.lower():
        return get_wikitext2(nsamples, seed, seqlen, tokenizer, eval_mode)
    if 'c4' in name.lower():
        return get_c4(nsamples, seed, seqlen, tokenizer, eval_mode)
    if 'ptb' in name.lower():
        return get_ptb(nsamples, seed, seqlen, tokenizer, eval_mode)
    
    raise ValueError(f"Unknown dataset: {name}")


def get_calibration_data(tokenizer, nsamples: int = 128, seqlen: int = 2048,
                         seed: int = 0, dataset: str = 'wikitext2') -> torch.Tensor:
    """
    Get calibration data as a tensor of input IDs.
    
    Args:
        tokenizer: Tokenizer to use
        nsamples: Number of samples
        seqlen: Sequence length
        seed: Random seed
        dataset: Dataset name
    
    Returns:
        Tensor of shape [nsamples, seqlen]
    """
    loader = get_loaders(dataset, nsamples, seed, seqlen, tokenizer, eval_mode=False)
    
    # Stack all samples
    samples = [item[0] for item in loader]  # Get input_ids from (input_ids, targets)
    data = torch.cat(samples, dim=0)  # [nsamples, seqlen]
    
    # Ensure valid token IDs
    vocab_size = len(tokenizer)
    data = torch.clamp(data, min=0, max=vocab_size - 1)
    
    return data
