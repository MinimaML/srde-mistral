#!/usr/bin/env python3
"""
SaRDinE Data Preparation Script

Creates a balanced 20B token dataset for 6-expert training:
- Math:     30% (6B tokens)
- Code:     30% (6B tokens)
- Science:  15% (3B tokens)
- Logic:    10% (2B tokens)
- Planning:  8% (1.5B tokens)
- Abstract:  7% (1.5B tokens)

Usage:
    python prepare_data.py --output_dir ./data --target_tokens 20e9
"""
import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Iterator
from datasets import load_dataset, Dataset, concatenate_datasets
from transformers import AutoTokenizer
from tqdm import tqdm
import random


# Domain configuration with target proportions
DOMAIN_CONFIG = {
    "math": {
        "weight": 0.30,
        "datasets": [
            ("nvidia/OpenMathInstruct-2", None, "train"),
            ("meta-math/MetaMathQA", None, "train"),
            ("microsoft/orca-math-word-problems-200k", None, "train"),
        ],
        "text_field": "text",  # Will be processed per dataset
    },
    "code": {
        "weight": 0.30,
        "datasets": [
            ("ise-uiuc/Magicoder-OSS-Instruct-75K", None, "train"),
            ("m-a-p/CodeFeedback-Filtered-Instruction", None, "train"),
        ],
        "text_field": "text",
    },
    "science": {
        "weight": 0.15,
        "datasets": [
            ("HuggingFaceFW/fineweb-edu", "sample-10BT", "train"),  # Will filter for science
        ],
        "text_field": "text",
        "filter_keywords": ["science", "physics", "chemistry", "biology", "experiment", 
                          "hypothesis", "scientific", "research", "study", "evidence"],
    },
    "logic": {
        "weight": 0.10,
        "datasets": [
            ("Rowan/hellaswag", None, "train"),
            ("tau/commonsense_qa", None, "train"),
        ],
        "text_field": "text",
    },
    "planning": {
        "weight": 0.08,
        "datasets": [
            ("hotpot_qa", "fullwiki", "train"),
        ],
        "text_field": "text",
    },
    "abstract": {
        "weight": 0.07,
        "datasets": [
            ("deepmind/aqua_rat", "raw", "train"),
            ("winogrande", "winogrande_xl", "train"),
        ],
        "text_field": "text",
    },
}


def count_tokens(text: str, tokenizer) -> int:
    """Count tokens in text."""
    return len(tokenizer.encode(text, add_special_tokens=False))


def format_sample(sample: Dict, dataset_name: str, domain: str) -> str:
    """Format a sample into training text based on dataset structure."""
    
    # Math datasets
    if "OpenMathInstruct" in dataset_name:
        question = sample.get("problem", sample.get("question", ""))
        answer = sample.get("generated_solution", sample.get("solution", ""))
        return f"Problem: {question}\n\nSolution: {answer}"
    
    elif "MetaMathQA" in dataset_name:
        return f"Question: {sample.get('query', '')}\n\nAnswer: {sample.get('response', '')}"
    
    elif "orca-math" in dataset_name:
        return f"Problem: {sample.get('question', '')}\n\nSolution: {sample.get('answer', '')}"
    
    # Code datasets
    elif "Magicoder" in dataset_name:
        return f"Task: {sample.get('problem', '')}\n\nCode:\n{sample.get('solution', '')}"
    
    elif "CodeFeedback" in dataset_name:
        return f"Query: {sample.get('query', '')}\n\nAnswer: {sample.get('answer', '')}"
    
    # Science (FineWeb-Edu)
    elif "fineweb" in dataset_name.lower():
        return sample.get("text", "")
    
    # Logic datasets
    elif "hellaswag" in dataset_name:
        ctx = sample.get("ctx", "")
        endings = sample.get("endings", [])
        label = sample.get("label", 0)
        if endings and 0 <= label < len(endings):
            return f"Context: {ctx}\n\nCompletion: {endings[label]}"
        return ctx
    
    elif "commonsense_qa" in dataset_name:
        q = sample.get("question", "")
        choices = sample.get("choices", {})
        answer_key = sample.get("answerKey", "")
        if choices and "text" in choices:
            texts = choices["text"]
            labels = choices.get("label", [])
            if answer_key in labels:
                idx = labels.index(answer_key)
                return f"Question: {q}\n\nAnswer: {texts[idx]}"
        return q
    
    # Planning datasets
    elif "hotpot" in dataset_name:
        q = sample.get("question", "")
        a = sample.get("answer", "")
        return f"Question: {q}\n\nAnswer: {a}"
    
    # Abstract datasets
    elif "aqua" in dataset_name:
        q = sample.get("question", "")
        rationale = sample.get("rationale", "")
        return f"Question: {q}\n\nReasoning: {rationale}"
    
    elif "winogrande" in dataset_name:
        sentence = sample.get("sentence", "")
        option1 = sample.get("option1", "")
        option2 = sample.get("option2", "")
        answer = sample.get("answer", "1")
        correct = option1 if answer == "1" else option2
        filled = sentence.replace("_", correct)
        return f"Sentence: {filled}"
    
    # Default
    return sample.get("text", str(sample))


def load_domain_data(domain: str, config: Dict, tokenizer, target_tokens: int) -> List[Dict]:
    """Load and format data for a domain until target tokens reached."""
    print(f"\n{'='*60}")
    print(f"Loading domain: {domain.upper()}")
    print(f"Target: {target_tokens/1e9:.1f}B tokens")
    print(f"{'='*60}")
    
    samples = []
    total_tokens = 0
    
    for dataset_name, subset, split in config["datasets"]:
        if total_tokens >= target_tokens:
            break
            
        print(f"\n  Loading: {dataset_name}")
        try:
            if subset:
                ds = load_dataset(dataset_name, subset, split=split, streaming=True)
            else:
                ds = load_dataset(dataset_name, split=split, streaming=True)
        except Exception as e:
            print(f"    ERROR: {e}")
            continue
        
        # Filter for science domain if needed
        filter_keywords = config.get("filter_keywords", None)
        
        pbar = tqdm(ds, desc=f"    Processing")
        for sample in pbar:
            if total_tokens >= target_tokens:
                break
            
            text = format_sample(sample, dataset_name, domain)
            
            # Apply keyword filter for science domain
            if filter_keywords:
                text_lower = text.lower()
                if not any(kw in text_lower for kw in filter_keywords):
                    continue
            
            if len(text) < 50:  # Skip very short samples
                continue
            
            tokens = count_tokens(text, tokenizer)
            
            samples.append({
                "text": text,
                "domain": domain,
                "tokens": tokens,
            })
            
            total_tokens += tokens
            pbar.set_postfix({"tokens": f"{total_tokens/1e9:.2f}B"})
    
    print(f"\n  Collected: {len(samples)} samples, {total_tokens/1e9:.2f}B tokens")
    return samples


def main():
    parser = argparse.ArgumentParser(description="Prepare SaRDinE training data")
    parser.add_argument("--output_dir", type=str, default="./data", help="Output directory")
    parser.add_argument("--target_tokens", type=float, default=20e9, help="Target total tokens")
    parser.add_argument("--tokenizer", type=str, default="mistralai/Ministral-3-14B-Reasoning-2512")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    random.seed(args.seed)
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("SaRDinE Data Preparation")
    print("=" * 60)
    print(f"Target tokens: {args.target_tokens/1e9:.0f}B")
    print(f"Output: {output_dir}")
    
    # Load tokenizer
    print(f"\nLoading tokenizer: {args.tokenizer}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    
    # Process each domain
    all_samples = []
    domain_stats = {}
    
    for domain, config in DOMAIN_CONFIG.items():
        target = int(args.target_tokens * config["weight"])
        samples = load_domain_data(domain, config, tokenizer, target)
        all_samples.extend(samples)
        domain_stats[domain] = {
            "samples": len(samples),
            "tokens": sum(s["tokens"] for s in samples),
            "weight": config["weight"],
        }
    
    # Shuffle
    print("\nShuffling data...")
    random.shuffle(all_samples)
    
    # Save
    print(f"\nSaving to {output_dir}...")
    
    # Save as JSONL
    jsonl_path = output_dir / "sardine_train.jsonl"
    with open(jsonl_path, "w") as f:
        for sample in tqdm(all_samples, desc="Writing JSONL"):
            f.write(json.dumps(sample) + "\n")
    
    # Save stats
    total_tokens = sum(s["tokens"] for s in all_samples)
    stats = {
        "total_samples": len(all_samples),
        "total_tokens": total_tokens,
        "domain_stats": domain_stats,
    }
    
    with open(output_dir / "stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    
    # Print summary
    print("\n" + "=" * 60)
    print("DATA PREPARATION COMPLETE")
    print("=" * 60)
    print(f"\nTotal samples: {len(all_samples):,}")
    print(f"Total tokens: {total_tokens/1e9:.2f}B")
    print(f"\nBy domain:")
    for domain, s in domain_stats.items():
        pct = s["tokens"] / total_tokens * 100
        print(f"  {domain:12} {s['tokens']/1e9:6.2f}B ({pct:5.1f}%) - {s['samples']:,} samples")
    
    print(f"\nOutput files:")
    print(f"  {jsonl_path}")
    print(f"  {output_dir / 'stats.json'}")


if __name__ == "__main__":
    main()
