#!/usr/bin/env python3
"""
SRDE Dataset Pipeline

Downloads domain datasets, tokenizes, and uploads to HuggingFace Hub.

Usage:
    python build_and_upload_dataset.py \
        --repo_name your-username/srde-mistral-dataset \
        --examples_per_domain 10000 \
        --hf_token YOUR_HF_TOKEN
"""
import argparse
import json
import os
import shutil
from pathlib import Path
from typing import List, Dict, Optional
import random
import struct

import numpy as np
from tqdm import tqdm


# Domain definitions - VERIFIED WORKING DATASETS
DOMAINS = {
    0: {
        "name": "math",
        "description": "Advanced Math",
        "datasets": [
            ("gsm8k", "main", "train"),
            ("hendrycks/competition_math", None, "train"),
            ("meta-math/MetaMathQA", None, "train"),
            ("microsoft/orca-math-word-problems-200k", None, "train"),
        ]
    },
    1: {
        "name": "logic", 
        "description": "Formal Logic",
        "datasets": [
            ("tasksource/bigbench", "logical_deduction_five_objects", "train"),
            ("tasksource/bigbench", "logical_deduction_seven_objects", "train"),
            ("tau/commonsense_qa", None, "train"),
            ("Rowan/hellaswag", None, "train"),
        ]
    },
    2: {
        "name": "code",
        "description": "Algorithm Design", 
        "datasets": [
            ("mbpp", "full", "train"),
            ("openai/openai_humaneval", None, "test"),
            ("m-a-p/CodeFeedback-Filtered-Instruction", None, "train"),
            ("flytech/python-codes-25k", None, "train"),
        ]
    },
    3: {
        "name": "science",
        "description": "Scientific Reasoning",
        "datasets": [
            ("allenai/sciq", None, "train"),
            ("allenai/ai2_arc", "ARC-Challenge", "train"),
            ("allenai/ai2_arc", "ARC-Easy", "train"),
            ("cais/mmlu", "college_physics", "test"),
            ("cais/mmlu", "college_chemistry", "test"),
        ]
    },
    4: {
        "name": "planning",
        "description": "Multi-step Planning",
        "datasets": [
            ("hotpot_qa", "distractor", "train"),
            ("tau/scrolls", "qasper", "train"),
            ("apple/DataCompLM-DCLM-baseline", None, "train"),
        ]
    },
    5: {
        "name": "abstract",
        "description": "Abstract/Symbolic",
        "datasets": [
            ("tasksource/bigbench", "abstract_narrative_understanding", "train"),
            ("tasksource/bigbench", "analogical_similarity", "train"),
            ("deepmind/aqua_rat", "raw", "train"),
            ("winogrande", "winogrande_xl", "train"),
        ]
    },
}


def parse_args():
    parser = argparse.ArgumentParser(description="Build and upload SRDE dataset")
    
    # Required
    parser.add_argument("--repo_name", type=str, required=True,
                       help="HuggingFace repo (e.g., username/srde-dataset)")
    
    # Optional
    parser.add_argument("--model_name", type=str, 
                       default="mistralai/Ministral-3-14B-Reasoning-2512")
    parser.add_argument("--examples_per_domain", type=int, default=10000)
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--hf_token", type=str, default=None,
                       help="HuggingFace token (or set HF_TOKEN env var)")
    parser.add_argument("--work_dir", type=str, default="./srde_dataset_build")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--private", action="store_true", 
                       help="Make repo private")
    parser.add_argument("--skip_download", action="store_true",
                       help="Skip download if data.jsonl exists")
    parser.add_argument("--skip_tokenize", action="store_true",
                       help="Skip tokenization if already done")
    
    return parser.parse_args()


def extract_text(example: dict, dataset_name: str) -> Optional[str]:
    """Extract reasoning text from various dataset formats."""
    text_parts = []
    
    # GSM8K
    if "gsm8k" in dataset_name.lower():
        if "question" in example and "answer" in example:
            return f"{example['question']}\n\n{example['answer']}"
    
    # MATH
    if "math" in dataset_name.lower():
        if "problem" in example and "solution" in example:
            return f"{example['problem']}\n\n{example['solution']}"
    
    # Code datasets
    if "code" in dataset_name.lower() or "apps" in dataset_name.lower():
        if "question" in example:
            text = example.get("question", example.get("problem", ""))
            if "solutions" in example and example["solutions"]:
                sols = example["solutions"]
                if isinstance(sols, str):
                    try:
                        sols = json.loads(sols)
                    except:
                        sols = [sols]
                if sols:
                    text += f"\n\n```python\n{sols[0]}\n```"
            return text if text else None
    
    # Generic fallback
    text_fields = ["question", "problem", "input", "prompt", "context", "text"]
    answer_fields = ["answer", "solution", "response", "output", "explanation"]
    
    for field in text_fields:
        if field in example and example[field]:
            text_parts.append(str(example[field]))
            break
    
    for field in answer_fields:
        if field in example and example[field]:
            val = example[field]
            if isinstance(val, list):
                val = val[0] if val else ""
            text_parts.append(str(val))
            break
    
    return "\n\n".join(text_parts) if text_parts else None


def download_datasets(args) -> str:
    """Step 1: Download and merge datasets."""
    from datasets import load_dataset
    
    output_file = Path(args.work_dir) / "data.jsonl"
    
    if args.skip_download and output_file.exists():
        print(f"[SKIP] Dataset already exists: {output_file}")
        return str(output_file)
    
    print("\n" + "="*60)
    print("Step 1: Downloading Datasets")
    print("="*60)
    
    random.seed(args.seed)
    all_examples = []
    
    for domain_id, domain in DOMAINS.items():
        print(f"\n[Domain {domain_id}] {domain['description']}")
        examples = []
        
        for dataset_name, subset, split in domain["datasets"]:
            if len(examples) >= args.examples_per_domain:
                break
            
            try:
                print(f"  Loading {dataset_name}...", end=" ", flush=True)
                
                if subset:
                    ds = load_dataset(dataset_name, subset, split=split)
                else:
                    ds = load_dataset(dataset_name, split=split)
                
                # Sample if too large
                remaining = args.examples_per_domain - len(examples)
                if len(ds) > remaining:
                    indices = random.sample(range(len(ds)), remaining)
                    ds = ds.select(indices)
                
                count = 0
                for ex in ds:
                    text = extract_text(ex, dataset_name)
                    if text and len(text) > 50:
                        examples.append({
                            "text": text,
                            "domain": domain["name"],
                            "domain_id": domain_id,
                            "source": dataset_name
                        })
                        count += 1
                        if len(examples) >= args.examples_per_domain:
                            break
                
                print(f"✓ ({count} examples)")
                
            except Exception as e:
                print(f"✗ ({e})")
                continue
        
        print(f"  Domain total: {len(examples)}")
        all_examples.extend(examples)
    
    # Shuffle and save
    random.shuffle(all_examples)
    
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        for ex in all_examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    
    print(f"\n✓ Saved {len(all_examples):,} examples to {output_file}")
    return str(output_file)


def tokenize_dataset(args, jsonl_path: str) -> str:
    """Step 2: Pre-tokenize the dataset."""
    from transformers import AutoTokenizer
    
    output_dir = Path(args.work_dir) / "tokenized"
    
    if args.skip_tokenize and (output_dir / "metadata.json").exists():
        print(f"[SKIP] Already tokenized: {output_dir}")
        return str(output_dir)
    
    print("\n" + "="*60)
    print("Step 2: Tokenizing Dataset")
    print("="*60)
    
    # Load tokenizer
    print(f"Loading tokenizer: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Count lines
    with open(jsonl_path, 'r') as f:
        total_lines = sum(1 for _ in f)
    
    # Tokenize
    chunk_size = 10000
    current_shard = []
    shard_idx = 0
    total_tokens = 0
    
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f, total=total_lines, desc="Tokenizing"):
            if not line.strip():
                continue
            
            try:
                data = json.loads(line)
                text = data.get("text", "")
                
                tokens = tokenizer.encode(
                    text,
                    max_length=args.max_length,
                    truncation=True,
                    add_special_tokens=True
                )
                
                current_shard.append(np.array(tokens, dtype=np.uint32))
                total_tokens += len(tokens)
                
            except Exception as e:
                continue
            
            # Write shard if full
            if len(current_shard) >= chunk_size:
                shard_path = output_dir / f"shard_{shard_idx:05d}.bin"
                with open(shard_path, 'wb') as sf:
                    sf.write(struct.pack('I', len(current_shard)))
                    for tokens in current_shard:
                        sf.write(struct.pack('I', len(tokens)))
                        sf.write(tokens.tobytes())
                shard_idx += 1
                current_shard = []
    
    # Write final shard
    if current_shard:
        shard_path = output_dir / f"shard_{shard_idx:05d}.bin"
        with open(shard_path, 'wb') as sf:
            sf.write(struct.pack('I', len(current_shard)))
            for tokens in current_shard:
                sf.write(struct.pack('I', len(tokens)))
                sf.write(tokens.tobytes())
        shard_idx += 1
    
    # Write metadata
    metadata = {
        "model_name": args.model_name,
        "max_length": args.max_length,
        "num_shards": shard_idx,
        "total_examples": total_lines,
        "total_tokens": total_tokens,
        "vocab_size": tokenizer.vocab_size
    }
    
    with open(output_dir / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Copy raw JSONL for reference
    shutil.copy(jsonl_path, output_dir / "data.jsonl")
    
    print(f"\n✓ Tokenized into {shard_idx} shards")
    print(f"  Total tokens: {total_tokens:,}")
    return str(output_dir)


def upload_to_hub(args, data_dir: str):
    """Step 3: Upload to HuggingFace Hub."""
    from huggingface_hub import HfApi, create_repo
    
    print("\n" + "="*60)
    print("Step 3: Uploading to HuggingFace Hub")
    print("="*60)
    
    # Get token
    hf_token = args.hf_token or os.environ.get("HF_TOKEN")
    if not hf_token:
        raise ValueError("HuggingFace token required. Use --hf_token or set HF_TOKEN env var")
    
    api = HfApi(token=hf_token)
    
    # Create repo if needed
    try:
        create_repo(
            repo_id=args.repo_name,
            repo_type="dataset",
            private=args.private,
            token=hf_token
        )
        print(f"✓ Created repo: {args.repo_name}")
    except Exception as e:
        if "already exists" in str(e).lower():
            print(f"✓ Repo exists: {args.repo_name}")
        else:
            raise
    
    # Upload all files
    print(f"\nUploading files from {data_dir}...")
    
    api.upload_folder(
        folder_path=data_dir,
        repo_id=args.repo_name,
        repo_type="dataset",
        token=hf_token
    )
    
    # Create README
    readme_content = f"""# SRDE Training Dataset

Pre-tokenized dataset for Sparse Routed Delta Experts (SRDE) training on Mistral-3-14B-Reasoning.

## Expert Domains

| ID | Domain | Description |
|----|--------|-------------|
| 0 | math | Advanced Math - algebra, calculus |
| 1 | logic | Formal Logic - proofs, deduction |
| 2 | code | Algorithm Design - optimization |
| 3 | science | Scientific Reasoning - physics, chemistry |
| 4 | planning | Multi-step Planning - decomposition |
| 5 | abstract | Abstract/Symbolic - patterns, analogy |

## Usage

```python
from huggingface_hub import hf_hub_download
import os

# Download all files
os.system(f"huggingface-cli download {args.repo_name} --local-dir ./data")

# Or use in training
python train.py --pretokenized_dir ./data --flash_attention --use_muon
```

## Statistics

- Model: `{args.model_name}`
- Max length: {args.max_length}
- Examples per domain: {args.examples_per_domain}
"""
    
    api.upload_file(
        path_or_fileobj=readme_content.encode(),
        path_in_repo="README.md",
        repo_id=args.repo_name,
        repo_type="dataset",
        token=hf_token
    )
    
    print(f"\n✓ Upload complete!")
    print(f"  Dataset URL: https://huggingface.co/datasets/{args.repo_name}")


def main():
    args = parse_args()
    
    print("="*60)
    print("SRDE Dataset Pipeline")
    print("="*60)
    print(f"Repo: {args.repo_name}")
    print(f"Examples per domain: {args.examples_per_domain}")
    print(f"Model: {args.model_name}")
    print("="*60)
    
    # Step 1: Download
    jsonl_path = download_datasets(args)
    
    # Step 2: Tokenize
    data_dir = tokenize_dataset(args, jsonl_path)
    
    # Step 3: Upload
    upload_to_hub(args, data_dir)
    
    print("\n" + "="*60)
    print("Pipeline Complete!")
    print("="*60)
    print(f"  Dataset: https://huggingface.co/datasets/{args.repo_name}")
    print(f"  Local copy: {data_dir}")
    print("\nTo train:")
    print(f"  huggingface-cli download {args.repo_name} --local-dir ./data")
    print(f"  python train.py --pretokenized_dir ./data --flash_attention --use_muon")


if __name__ == "__main__":
    main()
