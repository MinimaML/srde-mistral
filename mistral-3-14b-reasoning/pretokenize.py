#!/usr/bin/env python3
"""
Pre-tokenize dataset for faster training.

Converts JSONL text data to binary token files for zero-overhead data loading.

Usage:
    python pretokenize.py --input data.jsonl --output data_tokenized --model_name mistralai/Mistral-3-14B-Reasoning
"""
import argparse
import json
import os
from pathlib import Path
from typing import Optional
import struct

import numpy as np
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="Pre-tokenize dataset")
    parser.add_argument("--input", type=str, required=True, help="Input JSONL file")
    parser.add_argument("--output", type=str, required=True, help="Output directory")
    parser.add_argument("--model_name", type=str, default="mistralai/Mistral-3-14B-Reasoning")
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--chunk_size", type=int, default=10000, help="Examples per shard")
    return parser.parse_args()


def load_tokenizer(model_name: str):
    """Load tokenizer with caching."""
    from transformers import AutoTokenizer
    
    print(f"Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    return tokenizer


def tokenize_example(text: str, tokenizer, max_length: int) -> Optional[np.ndarray]:
    """Tokenize a single example."""
    try:
        tokens = tokenizer.encode(
            text,
            max_length=max_length,
            truncation=True,
            add_special_tokens=True
        )
        return np.array(tokens, dtype=np.uint32)
    except Exception as e:
        print(f"Error tokenizing: {e}")
        return None


def write_shard(tokens_list: list, output_path: Path, shard_idx: int):
    """Write a shard of tokenized data."""
    shard_path = output_path / f"shard_{shard_idx:05d}.bin"
    
    with open(shard_path, 'wb') as f:
        # Header: num_examples, then for each: length, tokens
        f.write(struct.pack('I', len(tokens_list)))
        
        for tokens in tokens_list:
            f.write(struct.pack('I', len(tokens)))
            f.write(tokens.tobytes())
    
    return shard_path


def main():
    args = parse_args()
    
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)
    
    tokenizer = load_tokenizer(args.model_name)
    
    print(f"Pre-tokenizing: {args.input}")
    print(f"Output directory: {output_path}")
    print(f"Max length: {args.max_length}")
    print(f"Chunk size: {args.chunk_size}")
    
    # Count lines first
    with open(args.input, 'r', encoding='utf-8') as f:
        total_lines = sum(1 for _ in f)
    
    print(f"Total examples: {total_lines:,}")
    
    # Tokenize
    current_shard = []
    shard_idx = 0
    total_tokens = 0
    
    with open(args.input, 'r', encoding='utf-8') as f:
        for line in tqdm(f, total=total_lines, desc="Tokenizing"):
            if not line.strip():
                continue
            
            try:
                data = json.loads(line)
                text = data.get("text", data.get("content", str(data)))
            except json.JSONDecodeError:
                continue
            
            tokens = tokenize_example(text, tokenizer, args.max_length)
            if tokens is not None:
                current_shard.append(tokens)
                total_tokens += len(tokens)
            
            # Write shard if full
            if len(current_shard) >= args.chunk_size:
                write_shard(current_shard, output_path, shard_idx)
                shard_idx += 1
                current_shard = []
    
    # Write final shard
    if current_shard:
        write_shard(current_shard, output_path, shard_idx)
        shard_idx += 1
    
    # Write metadata
    metadata = {
        "source": args.input,
        "model_name": args.model_name,
        "max_length": args.max_length,
        "num_shards": shard_idx,
        "total_examples": total_lines,
        "total_tokens": total_tokens,
        "vocab_size": tokenizer.vocab_size
    }
    
    with open(output_path / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n✓ Pre-tokenization complete!")
    print(f"  Shards: {shard_idx}")
    print(f"  Total tokens: {total_tokens:,}")
    print(f"  Avg tokens/example: {total_tokens / total_lines:.1f}")


if __name__ == "__main__":
    main()
