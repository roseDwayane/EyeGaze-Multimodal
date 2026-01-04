from __future__ import annotations
import os
import csv
import random
from typing import Dict, Optional, List
import numpy as np
from datasets import Dataset, DatasetDict

def read_eeg(path: str) -> np.ndarray:
    """Dispatch EEG file reading based on extension and ensure (C, T) layout."""
    ext = os.path.splitext(path)[1].lower()
    if ext == '.csv':
        with open(path, 'r', newline='') as f:
            reader = csv.reader(f)
            data = [row for row in reader]
        arr = np.asarray(data, dtype=np.float32)
    elif ext == '.npy':
        arr = np.load(path)
    else:
        raise ValueError(f'Unsupported EEG file extension: {ext} for path: {path}')
    if arr.ndim != 2:
        raise ValueError(f'EEG array must be 2D, got shape {arr.shape} from {path}')
    c, t = arr.shape
    if c > t:
        arr = arr.T
        c, t = arr.shape
    if c >= t:
        raise ValueError(f'Cannot determine (C, T) with C < T for {path}; got shape {arr.shape}')
    return arr

def gen_eeg(C: int=32, T: int=1024, *, sample_rate: float=256.0, mode: str='mixed', noise_std: float=0.1, num_components: int=3, seed: Optional[int]=None) -> np.ndarray:
    """
    No docstring provided.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(T, dtype=np.float32) / float(sample_rate)
    x = np.zeros((C, T), dtype=np.float32)
    if mode in {'sine', 'mixed'}:
        for c in range(C):
            freqs = rng.uniform(1.0, 40.0, size=(num_components,)).astype(np.float32)
            amps = rng.uniform(0.1, 1.0, size=(num_components,)).astype(np.float32)
            phases = rng.uniform(0.0, 2.0 * np.pi, size=(num_components,)).astype(np.float32)
            s = np.zeros_like(t)
            for f, a, p in zip(freqs, amps, phases):
                s += a * np.sin(2.0 * np.pi * f * t + p)
            x[c] += s.astype(np.float32)
    if mode in {'noise', 'mixed'}:
        x += rng.normal(loc=0.0, scale=noise_std, size=(C, T)).astype(np.float32)
    return x

def synthetic_eeg_generator(split_cfg: Dict, length: int, seed: Optional[int]=None):
    """
    A generator function that yields synthetic EEG samples.
    This is compatible with `datasets.Dataset.from_generator`.
    """
    C = int(split_cfg.get('C', 30))
    T = int(split_cfg.get('T', 1024))
    sample_rate = float(split_cfg.get('sample_rate', 256.0))
    spec_target = split_cfg.get('target', {})
    spec_attr = split_cfg.get('attr', {})
    for i in range(length):
        s_attr = None if seed is None else seed * 100003 + i
        s_tgt = None if seed is None else seed * 100019 + i
        input_values = gen_eeg(C=C, T=T, sample_rate=sample_rate, seed=s_attr, **spec_attr)
        labels = gen_eeg(C=C, T=T, sample_rate=sample_rate, seed=s_tgt, **spec_target)
        yield {'input_values': input_values.astype(np.float32), 'labels': labels.astype(np.float32)}

def real_eeg_generator(split_name: str, data_cfg: Dict, seed: Optional[int]=None):
    """
    A generator that yields real EEG samples from files.
    Mimics the behavior of the original EEGDataset.
    """
    root = data_cfg.get('root')
    splits_cfg = data_cfg.get('splits', {})
    rng = random.Random(seed)
    categories = ['Brain', 'ChannelNoise', 'Eye', 'Heart', 'LineNoise', 'Muscle', 'Other']
    base_dir = os.path.join(root, split_name)
    brain_dir = os.path.join(base_dir, 'Brain')
    if split_name in splits_cfg and isinstance(splits_cfg[split_name], list):
        files: List[str] = splits_cfg[split_name]
    else:
        files = sorted(os.listdir(brain_dir))
    for fname in files:
        brain_path = os.path.join(brain_dir, fname)
        if not os.path.isfile(brain_path):
            continue
        category = rng.choice(categories)
        noise_path = os.path.join(base_dir, category, fname)
        labels = read_eeg(brain_path)
        if os.path.isfile(noise_path):
            input_values = read_eeg(noise_path)
        else:
            input_values = labels.copy()
        yield {'input_values': input_values.astype(np.float32), 'labels': labels.astype(np.float32)}

def build_hf_datasets(config: dict, seed: Optional[int]=42) -> DatasetDict:
    """
    Factory that builds a Hugging Face DatasetDict for either real or synthetic data.

    Heuristic: If `data.root` is a valid directory for real data, it will be used.
    Otherwise, it falls back to generating synthetic data based on the config.
    """
    data_cfg = config.get('data', {})
    root = data_cfg.get('root')
    ds_dict = DatasetDict()
    use_real_data = False
    if isinstance(root, str):
        if os.path.isdir(os.path.join(root, 'train', 'Brain')):
            use_real_data = True
    for split_name in ['train', 'val', 'test']:
        if use_real_data:
            print(f"Building '{split_name}' split from real data source: {root}")
            ds_dict[split_name] = Dataset.from_generator(real_eeg_generator, gen_kwargs={'split_name': split_name, 'data_cfg': data_cfg, 'seed': seed})
        else:
            print(f"Building '{split_name}' split from synthetic data generator.")
            splits_cfg = data_cfg.get('splits', {})
            split_params = splits_cfg.get(split_name, {})
            length = split_params.get('length', 1000 if split_name == 'train' else 100)
            ds_dict[split_name] = Dataset.from_generator(synthetic_eeg_generator, gen_kwargs={'split_cfg': split_params, 'length': length, 'seed': seed})
    return ds_dict