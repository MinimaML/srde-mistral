"""
Pre-tokenized Dataset Loader

Zero-overhead data loading from pre-tokenized binary shards.
"""
import os
import json
import struct
import random
from pathlib import Path
from typing import Optional, List

import numpy as np
import torch
from torch.utils.data import Dataset, IterableDataset


class PreTokenizedDataset(Dataset):
    """
    Map-style dataset for pre-tokenized data.
    
    Loads all data into memory - best for datasets that fit in RAM.
    """
    
    def __init__(self, data_dir: str, max_length: int = 2048):
        self.data_dir = Path(data_dir)
        self.max_length = max_length
        
        # Load metadata
        with open(self.data_dir / "metadata.json") as f:
            self.metadata = json.load(f)
        
        print(f"Loading pre-tokenized data from {data_dir}")
        print(f"  Total examples: {self.metadata['total_examples']:,}")
        print(f"  Total tokens: {self.metadata['total_tokens']:,}")
        
        # Load all shards
        self.examples = []
        for shard_idx in range(self.metadata['num_shards']):
            shard_path = self.data_dir / f"shard_{shard_idx:05d}.bin"
            self._load_shard(shard_path)
        
        print(f"  Loaded {len(self.examples):,} examples")
    
    def _load_shard(self, shard_path: Path):
        """Load a single shard."""
        with open(shard_path, 'rb') as f:
            num_examples = struct.unpack('I', f.read(4))[0]
            
            for _ in range(num_examples):
                length = struct.unpack('I', f.read(4))[0]
                tokens = np.frombuffer(f.read(length * 4), dtype=np.uint32)
                self.examples.append(tokens)
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        tokens = self.examples[idx]
        
        # Pad or truncate
        if len(tokens) > self.max_length:
            tokens = tokens[:self.max_length]
        elif len(tokens) < self.max_length:
            tokens = np.pad(tokens, (0, self.max_length - len(tokens)))
        
        input_ids = torch.tensor(tokens, dtype=torch.long)
        attention_mask = (input_ids != 0).long()
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": input_ids.clone()
        }


class StreamingPreTokenizedDataset(IterableDataset):
    """
    Iterable dataset for pre-tokenized data.
    
    Streams from disk - best for very large datasets.
    """
    
    def __init__(
        self,
        data_dir: str,
        max_length: int = 2048,
        shuffle: bool = True,
        seed: int = 42
    ):
        self.data_dir = Path(data_dir)
        self.max_length = max_length
        self.shuffle = shuffle
        self.seed = seed
        
        with open(self.data_dir / "metadata.json") as f:
            self.metadata = json.load(f)
        
        self.shard_paths = sorted(self.data_dir.glob("shard_*.bin"))
    
    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        
        shards = list(self.shard_paths)
        
        if self.shuffle:
            random.seed(self.seed)
            random.shuffle(shards)
        
        # Distribute shards across workers
        if worker_info is not None:
            shards = shards[worker_info.id::worker_info.num_workers]
        
        for shard_path in shards:
            yield from self._iterate_shard(shard_path)
    
    def _iterate_shard(self, shard_path: Path):
        """Iterate through a shard."""
        examples = []
        
        with open(shard_path, 'rb') as f:
            num_examples = struct.unpack('I', f.read(4))[0]
            
            for _ in range(num_examples):
                length = struct.unpack('I', f.read(4))[0]
                tokens = np.frombuffer(f.read(length * 4), dtype=np.uint32).copy()
                examples.append(tokens)
        
        if self.shuffle:
            random.shuffle(examples)
        
        for tokens in examples:
            # Pad or truncate
            if len(tokens) > self.max_length:
                tokens = tokens[:self.max_length]
            elif len(tokens) < self.max_length:
                tokens = np.pad(tokens, (0, self.max_length - len(tokens)))
            
            input_ids = torch.tensor(tokens, dtype=torch.long)
            attention_mask = (input_ids != 0).long()
            
            yield {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": input_ids.clone()
            }


def create_dataloader(
    data_dir: str,
    batch_size: int = 1,
    max_length: int = 2048,
    num_workers: int = 4,
    streaming: bool = False,
    shuffle: bool = True
):
    """
    Create a DataLoader for pre-tokenized data.
    
    Args:
        data_dir: Path to pre-tokenized data directory
        batch_size: Batch size
        max_length: Max sequence length
        num_workers: Number of data loading workers
        streaming: Use streaming (IterableDataset) vs map-style
        shuffle: Shuffle data
    """
    from torch.utils.data import DataLoader
    
    if streaming:
        dataset = StreamingPreTokenizedDataset(
            data_dir, max_length=max_length, shuffle=shuffle
        )
        return DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=True
        )
    else:
        dataset = PreTokenizedDataset(data_dir, max_length=max_length)
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True
        )
