import random
import os
import logging
import json
import glob as _glob

# Cache directories: HF_HUB_CACHE is the actual hub directory (HF_HOME/hub)
# where models--org--name/ folders live.  HF_DATASETS_CACHE is for datasets.
# Both are set by run_quantize.py before any imports.
_hf_home = os.environ.get('HF_HOME', '/root/autodl-tmp/huggingface')
MODEL_CACHE_DIR = os.environ.get('HF_HUB_CACHE', os.path.join(_hf_home, 'hub'))
DATASETS_CACHE_DIR = os.environ.get('HF_DATASETS_CACHE', '/root/autodl-tmp/datasets')

from transformers import AutoTokenizer
from datasets import load_dataset, load_from_disk, Dataset

# Legacy modelscope model paths (backward compat for old downloads)
MODEL_NAME_MAPPING = {
    'meta-llama/Llama-2-7b-hf': '/root/autodl-tmp/models/shakechen/Llama-2-7b-hf',
    'meta-llama/Llama-2-13b-hf': '/root/autodl-tmp/models/shakechen/Llama-2-13b-hf',
    'meta-llama/Llama-2-70b-hf': '/root/autodl-tmp/models/shakechen/Llama-2-70b-hf',
    'meta-llama/Meta-Llama-3-8B': '/root/autodl-tmp/models/LLM-Research/Meta-Llama-3-8B',
    'meta-llama/Meta-Llama-3-70B': '/root/autodl-tmp/models/LLM-Research/Meta-Llama-3-70B',
    'shakechen/Llama-2-7b-hf': '/root/autodl-tmp/models/shakechen/Llama-2-7b-hf',
    'shakechen/Llama-2-13b-hf': '/root/autodl-tmp/models/shakechen/Llama-2-13b-hf',
    'shakechen/Llama-2-70b-hf': '/root/autodl-tmp/models/shakechen/Llama-2-70b-hf',
}

# Legacy hardcoded dataset paths (fallback if load_dataset fails)
_LEGACY_DATASET_PATHS = {
    'wikitext2': '/root/autodl-tmp/datasets/wikitext/wikitext-2-v1/1.0.0/6280e5a53c82b20da4f99f484fa6f0ca9de738ff12f59efb0815fe7d8ae21478',
    'ptb': '/root/autodl-tmp/datasets/xeon09112___ptb_text_only/default-ce1f658bdfd12953/0.0.0/master',
}

# HF dataset identifiers (matching stat_and_download.py)
_HF_DATASET_INFO = {
    'wikitext2': {'repo': 'wikitext', 'config': 'wikitext-2-raw-v1', 'text_key': 'text'},
    'ptb':       {'repo': 'ptb-text-only', 'config': 'penn_treebank', 'text_key': 'sentence'},
    'c4':        {'repo': 'allenai/c4', 'config': None, 'text_key': 'text'},
}


def convert_model_name(model_name):
    """Convert model name to local path if available."""
    if model_name in MODEL_NAME_MAPPING:
        local_path = MODEL_NAME_MAPPING[model_name]
        if os.path.exists(local_path):
            logging.info(f'Using local model: {model_name} -> {local_path}')
            return local_path
    return model_name


def _load_hf_dataset(dataset_name, split):
    """Load dataset via HuggingFace datasets library (uses HF_DATASETS_CACHE)."""
    info = _HF_DATASET_INFO.get(dataset_name)
    if info is None:
        raise ValueError(f'Unknown HF dataset: {dataset_name}')

    kwargs = dict(
        split=split,
        cache_dir=DATASETS_CACHE_DIR,
        trust_remote_code=True,
    )

    # C4 needs special data_files handling (same as stat_and_download.py)
    if dataset_name == 'c4':
        if split == 'validation':
            kwargs['data_files'] = {
                'validation': ['en/c4-validation.00000-of-00008.json.gz']
            }
        loader = lambda **kw: load_dataset(info['repo'], **kw)
    elif info['config']:
        loader = lambda **kw: load_dataset(info['repo'], info['config'], **kw)
    else:
        loader = lambda **kw: load_dataset(info['repo'], **kw)

    # Try with network first, then force offline mode to use cache only
    try:
        return loader(**kwargs)
    except Exception:
        pass

    # Retry with HF_DATASETS_OFFLINE to avoid any network calls
    old_val = os.environ.get('HF_DATASETS_OFFLINE')
    try:
        os.environ['HF_DATASETS_OFFLINE'] = '1'
        return loader(**kwargs)
    finally:
        if old_val is None:
            os.environ.pop('HF_DATASETS_OFFLINE', None)
        else:
            os.environ['HF_DATASETS_OFFLINE'] = old_val


def _load_legacy_dataset(dataset_name, split):
    """Fallback: load from legacy hardcoded paths."""
    if dataset_name not in _LEGACY_DATASET_PATHS:
        raise ValueError(f'Unknown dataset: {dataset_name}')

    base_path = _LEGACY_DATASET_PATHS[dataset_name]
    if not os.path.exists(base_path):
        raise FileNotFoundError(f'Dataset directory not found: {base_path}')

    try:
        ds = load_from_disk(base_path)
        if split in ds:
            return ds[split]
        return ds
    except Exception as e:
        logging.warning(f"load_from_disk failed: {e}")

    # Try arrow files
    files = os.listdir(base_path)
    arrow_files = [f for f in files if f.endswith('.arrow') and split in f]
    if arrow_files:
        try:
            return Dataset.from_file(os.path.join(base_path, arrow_files[0]))
        except Exception:
            pass

    # Try parquet files
    parquet_files = [f for f in files if f.endswith('.parquet') and split in f]
    if parquet_files:
        import pyarrow.parquet as pq
        all_data = {}
        for pq_file in sorted(parquet_files):
            table = pq.read_table(os.path.join(base_path, pq_file))
            data = table.to_pydict()
            if not all_data:
                all_data = data
            else:
                for key in data:
                    all_data[key].extend(data[key])
        return all_data

    # Try json files
    json_files = [f for f in files if f.endswith('.json') and 'dataset_info' not in f]
    data_json = [f for f in json_files if split in f] or json_files
    if data_json:
        with open(os.path.join(base_path, data_json[0]), 'r') as f:
            return json.load(f)

    raise FileNotFoundError(f'No {split} data found in {base_path}')


def _scan_hf_cache_for_dataset(dataset_name, split):
    """Last-resort: scan HF datasets cache for arrow files matching the dataset."""
    info = _HF_DATASET_INFO.get(dataset_name)
    if info is None:
        raise ValueError(f'Unknown dataset: {dataset_name}')

    # Build search patterns based on the HF repo name
    # HF datasets caches under: {cache_dir}/{repo_org}___{repo_name}/{config_hash}/...
    repo = info['repo'].replace('/', '___')
    search_dirs = [DATASETS_CACHE_DIR]
    hf_home = os.environ.get('HF_HOME', '')
    if hf_home and hf_home != DATASETS_CACHE_DIR:
        search_dirs.append(hf_home)

    for base in search_dirs:
        if not os.path.isdir(base):
            continue
        # Look for directories matching the repo name
        for pattern in [f'{repo}*', f'*{dataset_name}*']:
            for ds_dir in _glob.glob(os.path.join(base, pattern)):
                if not os.path.isdir(ds_dir):
                    continue
                # Try load_from_disk on the directory itself
                try:
                    ds = load_from_disk(ds_dir)
                    if hasattr(ds, 'keys') and split in ds:
                        logging.info(f'Loaded {dataset_name}/{split} via cache scan: {ds_dir}')
                        return ds[split]
                    elif not hasattr(ds, 'keys'):
                        logging.info(f'Loaded {dataset_name} via cache scan: {ds_dir}')
                        return ds
                except Exception:
                    pass
                # Walk subdirectories for arrow files
                for root, dirs, files in os.walk(ds_dir):
                    arrow_files = [f for f in files if f.endswith('.arrow') and split in f]
                    if arrow_files:
                        try:
                            ds = Dataset.from_file(os.path.join(root, arrow_files[0]))
                            logging.info(f'Loaded {dataset_name}/{split} from arrow: {root}')
                            return ds
                        except Exception:
                            pass
                    # Try load_from_disk on subdirectories that look like dataset dirs
                    if 'dataset_info.json' in files:
                        try:
                            ds = load_from_disk(root)
                            if hasattr(ds, 'keys') and split in ds:
                                logging.info(f'Loaded {dataset_name}/{split} from subdir: {root}')
                                return ds[split]
                        except Exception:
                            pass

    raise FileNotFoundError(
        f'Could not find {dataset_name}/{split} in HF cache directories: {search_dirs}'
    )


def load_local_dataset(dataset_name, split='train'):
    """Load dataset: try HF cache first, then legacy paths, then cache scan."""
    # Try HF datasets cache first (datasets downloaded by stat_and_download.py)
    try:
        ds = _load_hf_dataset(dataset_name, split)
        logging.info(f'Loaded {dataset_name}/{split} from HF datasets cache')
        return ds
    except Exception as e:
        logging.info(f'HF cache load failed for {dataset_name}/{split}: {e}')

    # Fallback to legacy hardcoded paths
    if dataset_name in _LEGACY_DATASET_PATHS:
        try:
            logging.info(f'Trying legacy path for {dataset_name}/{split}...')
            return _load_legacy_dataset(dataset_name, split)
        except Exception as e:
            logging.info(f'Legacy load failed for {dataset_name}/{split}: {e}')

    # Last resort: scan the HF cache directory for arrow files
    logging.info(f'Scanning HF cache for {dataset_name}/{split}...')
    return _scan_hf_cache_for_dataset(dataset_name, split)


def _get_texts(data, text_key='text'):
    """Extract text column from various dataset formats."""
    if hasattr(data, 'column_names'):
        return data[text_key]
    elif isinstance(data, dict):
        return data[text_key]
    else:
        return [item[text_key] for item in data]


def get_wikitext2(nsamples, seed, seqlen, model, hf_token, eval_mode=False):

    model = convert_model_name(model)
    tokenizer = AutoTokenizer.from_pretrained(
        model, use_fast=False, trust_remote_code=True,
        cache_dir=MODEL_CACHE_DIR, token=hf_token,
    )

    if eval_mode:
        data = load_local_dataset('wikitext2', 'test')
        texts = _get_texts(data, 'text')
        testenc = tokenizer("\n\n".join(texts), return_tensors='pt')
        return testenc
    else:
        data = load_local_dataset('wikitext2', 'train')
        texts = _get_texts(data, 'text')
        trainenc = tokenizer("\n\n".join(texts), return_tensors='pt')
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


def get_c4_new(nsamples, seed, seqlen, model, hf_token=None, eval_mode=False):

    model = convert_model_name(model)
    tokenizer = AutoTokenizer.from_pretrained(
        model, use_fast=False, trust_remote_code=True,
        cache_dir=MODEL_CACHE_DIR, token=hf_token,
    )

    if eval_mode:
        data = load_local_dataset('c4', 'validation')
        texts = _get_texts(data, 'text')
        testenc = tokenizer("\n\n".join(texts), return_tensors='pt')
        return testenc
    else:
        data = load_local_dataset('c4', 'validation')
        texts = _get_texts(data, 'text')
        trainenc = tokenizer("\n\n".join(texts), return_tensors='pt')
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


def get_ptb_new(nsamples, seed, seqlen, model, hf_token, eval_mode=False):

    model = convert_model_name(model)
    tokenizer = AutoTokenizer.from_pretrained(
        model, use_fast=False, trust_remote_code=True,
        cache_dir=MODEL_CACHE_DIR, token=hf_token,
    )

    if eval_mode:
        data = load_local_dataset('ptb', 'test')
        sentences = _get_texts(data, 'sentence')
        testenc = tokenizer(" ".join(sentences), return_tensors='pt')
        return testenc
    else:
        data = load_local_dataset('ptb', 'train')
        sentences = _get_texts(data, 'sentence')
        trainenc = tokenizer(" ".join(sentences), return_tensors='pt')
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


def get_loaders(
    name, nsamples=128, seed=0, seqlen=2048, model='', hf_token=None, eval_mode=False
):
    if 'wikitext2' in name:
        return get_wikitext2(nsamples, seed, seqlen, model, hf_token, eval_mode)
    if 'ptb' in name:
        return get_ptb_new(nsamples, seed, seqlen, model, hf_token, eval_mode)
    if 'c4' in name:
        return get_c4_new(nsamples, seed, seqlen, model, hf_token, eval_mode)
