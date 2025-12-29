#!/usr/bin/env python3
"""
SRDE Dataset Builder

Downloads and prepares domain-specific datasets for 6-expert SRDE training.

Expert Domains:
    0: Advanced Math - GSM8K, MATH, MetaMathQA
    1: Formal Logic - LogiQA, ReClor, FOLIO
    2: Algorithm Design - CodeContests, APPS, LeetCode
    3: Scientific Reasoning - SciQ, ARC, GPQA
    4: Multi-step Planning - StrategyQA, HotpotQA
    5: Abstract/Symbolic - BIG-Bench Hard, AQuA-RAT

Usage:
    python build_dataset.py --output data.jsonl --examples_per_domain 10000
"""
import argparse
import json
import os
from pathlib import Path
from typing import List, Dict, Optional
import random

from tqdm import tqdm


# Domain definitions
DOMAINS = {
    0: {
        "name": "math",
        "description": "Advanced Math",
        "datasets": [
            ("gsm8k", "main", "train"),
            ("lighteval/MATH", "all", "train"),
            ("meta-math/MetaMathQA", None, "train"),
        ]
    },
    1: {
        "name": "logic",
        "description": "Formal Logic",
        "datasets": [
            ("lucasmccabe/logiqa", None, "train"),
            ("metaeval/reclor", None, "train"),
        ]
    },
    2: {
        "name": "code",
        "description": "Algorithm Design",
        "datasets": [
            ("deepmind/code_contests", None, "train"),
            ("codeparrot/apps", "all", "train"),
        ]
    },
    3: {
        "name": "science",
        "description": "Scientific Reasoning",
        "datasets": [
            ("allenai/sciq", None, "train"),
            ("allenai/ai2_arc", "ARC-Challenge", "train"),
        ]
    },
    4: {
        "name": "planning",
        "description": "Multi-step Planning",
        "datasets": [
            ("wics/strategy-qa", None, "train"),
            ("hotpot_qa", "fullwiki", "train"),
        ]
    },
    5: {
        "name": "abstract",
        "description": "Abstract/Symbolic",
        "datasets": [
            ("maveriq/bigbenchhard", None, "train"),
            ("deepmind/aqua_rat", None, "train"),
        ]
    },
}


def parse_args():
    parser = argparse.ArgumentParser(description="Build SRDE training dataset")
    parser.add_argument("--output", type=str, default="data.jsonl", help="Output file")
    parser.add_argument("--examples_per_domain", type=int, default=10000)
    parser.add_argument("--domains", type=str, default="all", 
                       help="Comma-separated domain IDs or 'all'")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cache_dir", type=str, default=None)
    return parser.parse_args()


def extract_text(example: dict, dataset_name: str) -> Optional[str]:
    """Extract reasoning text from various dataset formats."""
    
    # Try common field names
    text_fields = [
        "question", "problem", "input", "prompt", 
        "context", "passage", "text"
    ]
    answer_fields = [
        "answer", "solution", "response", "output",
        "explanation", "rationale", "chain_of_thought"
    ]
    
    text_parts = []
    
    # Get question/problem
    for field in text_fields:
        if field in example and example[field]:
            text_parts.append(str(example[field]))
            break
    
    # Get answer/solution
    for field in answer_fields:
        if field in example and example[field]:
            val = example[field]
            if isinstance(val, list):
                val = val[0] if val else ""
            text_parts.append(str(val))
            break
    
    # Special handling for specific datasets
    if "gsm8k" in dataset_name.lower():
        if "question" in example and "answer" in example:
            return f"{example['question']}\n\n{example['answer']}"
    
    if "math" in dataset_name.lower():
        if "problem" in example and "solution" in example:
            return f"{example['problem']}\n\n{example['solution']}"
    
    if "code" in dataset_name.lower() or "apps" in dataset_name.lower():
        if "question" in example:
            text = example["question"]
            if "solutions" in example and example["solutions"]:
                sols = example["solutions"]
                if isinstance(sols, str):
                    try:
                        sols = json.loads(sols)
                    except:
                        sols = [sols]
                if sols:
                    text += f"\n\n```python\n{sols[0]}\n```"
            return text
    
    if text_parts:
        return "\n\n".join(text_parts)
    
    return None


def load_domain_data(domain_id: int, max_examples: int, cache_dir: Optional[str]) -> List[dict]:
    """Load data for a specific domain."""
    from datasets import load_dataset
    
    domain = DOMAINS[domain_id]
    examples = []
    
    print(f"\n[Domain {domain_id}] {domain['description']}")
    
    for dataset_name, subset, split in domain["datasets"]:
        if len(examples) >= max_examples:
            break
        
        try:
            print(f"  Loading {dataset_name}...", end=" ", flush=True)
            
            kwargs = {"cache_dir": cache_dir} if cache_dir else {}
            if subset:
                ds = load_dataset(dataset_name, subset, split=split, **kwargs)
            else:
                ds = load_dataset(dataset_name, split=split, **kwargs)
            
            # Sample if too large
            if len(ds) > max_examples - len(examples):
                indices = random.sample(range(len(ds)), max_examples - len(examples))
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
                    
                    if len(examples) >= max_examples:
                        break
            
            print(f"✓ ({count} examples)")
            
        except Exception as e:
            print(f"✗ ({e})")
            continue
    
    print(f"  Total: {len(examples)} examples")
    return examples


def main():
    args = parse_args()
    random.seed(args.seed)
    
    print("="*60)
    print("SRDE Dataset Builder")
    print("="*60)
    
    # Determine domains to process
    if args.domains == "all":
        domain_ids = list(DOMAINS.keys())
    else:
        domain_ids = [int(d) for d in args.domains.split(",")]
    
    print(f"Domains: {domain_ids}")
    print(f"Examples per domain: {args.examples_per_domain}")
    print(f"Output: {args.output}")
    
    # Load all domain data
    all_examples = []
    
    for domain_id in domain_ids:
        examples = load_domain_data(
            domain_id, 
            args.examples_per_domain,
            args.cache_dir
        )
        all_examples.extend(examples)
    
    # Shuffle
    random.shuffle(all_examples)
    
    # Write output
    print(f"\nWriting {len(all_examples)} examples to {args.output}...")
    
    with open(args.output, 'w', encoding='utf-8') as f:
        for ex in tqdm(all_examples, desc="Writing"):
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    
    # Summary
    print("\n" + "="*60)
    print("Dataset Summary")
    print("="*60)
    
    domain_counts = {}
    for ex in all_examples:
        d = ex["domain"]
        domain_counts[d] = domain_counts.get(d, 0) + 1
    
    for domain, count in sorted(domain_counts.items()):
        pct = 100 * count / len(all_examples)
        print(f"  {domain:15} {count:6,} ({pct:5.1f}%)")
    
    print(f"\n  Total:          {len(all_examples):,}")
    print("="*60)


if __name__ == "__main__":
    main()
