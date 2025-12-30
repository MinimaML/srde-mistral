"""
SRDE Evaluation Script

Evaluate trained SRDE model on reasoning benchmarks and compare to baseline.
"""
import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Any

import torch
from tqdm import tqdm
from transformers import AutoTokenizer

from srde import create_srde_model, SRDEModel
from config import SRDEConfig


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate SRDE model")
    
    parser.add_argument("--model_name", type=str, default="mistralai/Ministral-3-14B-Reasoning-2512")
    parser.add_argument("--checkpoint", type=str, required=True, help="SRDE checkpoint path")
    parser.add_argument("--eval_file", type=str, required=True, help="JSONL evaluation file")
    parser.add_argument("--output_file", type=str, default="eval_results.json")
    parser.add_argument("--max_samples", type=int, default=1000)
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--bf16", action="store_true", default=True)
    parser.add_argument("--compare_baseline", action="store_true", 
                       help="Also evaluate baseline model without SRDE")
    
    return parser.parse_args()


def load_eval_data(file_path: str, max_samples: int) -> List[Dict]:
    """Load evaluation data from JSONL."""
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            if line.strip():
                try:
                    item = json.loads(line)
                    data.append(item)
                    if len(data) >= max_samples:
                        break
                except json.JSONDecodeError:
                    continue
    return data


def compute_perplexity(
    model: torch.nn.Module,
    tokenizer,
    texts: List[str],
    max_length: int,
    device: str = "cuda"
) -> float:
    """Compute average perplexity on texts."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    
    with torch.no_grad():
        for text in tqdm(texts, desc="Computing perplexity"):
            encodings = tokenizer(
                text,
                truncation=True,
                max_length=max_length,
                return_tensors="pt"
            )
            
            input_ids = encodings["input_ids"].to(device)
            
            if input_ids.shape[1] < 2:
                continue
            
            outputs = model(input_ids)
            logits = outputs.get('logits') if isinstance(outputs, dict) else outputs.logits
            
            if logits is None:
                continue
            
            # Compute loss
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = input_ids[:, 1:].contiguous()
            
            loss_fct = torch.nn.CrossEntropyLoss(reduction='sum')
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )
            
            total_loss += loss.item()
            total_tokens += shift_labels.numel()
    
    avg_loss = total_loss / total_tokens if total_tokens > 0 else float('inf')
    perplexity = torch.exp(torch.tensor(avg_loss)).item()
    
    return perplexity


def compute_expert_usage(model: SRDEModel) -> Dict[str, Any]:
    """Analyze expert usage patterns from SRDE layers."""
    usage_stats = {}
    
    for layer_key, srde_layer in model.srde_layers.items():
        # Get last routing stats if available
        if hasattr(srde_layer, '_last_selected_experts'):
            selected = srde_layer._last_selected_experts
            if selected is not None:
                # Count expert usage
                usage = torch.bincount(
                    selected.flatten(),
                    minlength=len(srde_layer.experts)
                )
                usage_stats[f"layer_{layer_key}"] = {
                    "expert_usage": usage.tolist(),
                    "unlocked_experts": list(srde_layer._unlocked_experts)
                }
    
    return usage_stats


def evaluate_by_domain(
    model: torch.nn.Module,
    tokenizer,
    data: List[Dict],
    max_length: int,
    device: str = "cuda"
) -> Dict[str, float]:
    """Evaluate perplexity per domain."""
    domain_texts = {}
    
    for item in data:
        domain = item.get("domain", "unknown")
        text = item.get("text", "")
        if domain not in domain_texts:
            domain_texts[domain] = []
        domain_texts[domain].append(text)
    
    domain_perplexities = {}
    for domain, texts in domain_texts.items():
        if texts:
            ppl = compute_perplexity(model, tokenizer, texts[:100], max_length, device)
            domain_perplexities[domain] = ppl
            print(f"  {domain}: {ppl:.2f}")
    
    return domain_perplexities


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print("=" * 60)
    print("SRDE Evaluation")
    print("=" * 60)
    
    # Load tokenizer
    print("\n[1/4] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load SRDE model
    print("\n[2/4] Loading SRDE model...")
    dtype = torch.bfloat16 if args.bf16 else torch.float32
    model = create_srde_model(
        model_name=args.model_name,
        torch_dtype=dtype
    )
    
    # Load checkpoint
    checkpoint_path = Path(args.checkpoint) / "srde_weights.pt"
    if checkpoint_path.exists():
        model.load_srde_weights(str(checkpoint_path))
        print(f"  Loaded SRDE weights from {checkpoint_path}")
    else:
        print(f"  WARNING: No SRDE weights found at {checkpoint_path}")
    
    # Load eval data
    print("\n[3/4] Loading evaluation data...")
    data = load_eval_data(args.eval_file, args.max_samples)
    texts = [item.get("text", "") for item in data if item.get("text")]
    print(f"  Loaded {len(texts)} examples")
    
    # Evaluate
    print("\n[4/4] Running evaluation...")
    results = {}
    
    # Overall perplexity
    print("\nOverall perplexity:")
    srde_ppl = compute_perplexity(model, tokenizer, texts, args.max_length, device)
    print(f"  SRDE model: {srde_ppl:.2f}")
    results["srde_perplexity"] = srde_ppl
    
    # Per-domain perplexity
    print("\nPer-domain perplexity:")
    domain_ppl = evaluate_by_domain(model, tokenizer, data, args.max_length, device)
    results["domain_perplexities"] = domain_ppl
    
    # Expert usage
    print("\nExpert usage analysis:")
    expert_usage = compute_expert_usage(model)
    results["expert_usage"] = expert_usage
    for layer, stats in expert_usage.items():
        print(f"  {layer}: {stats['expert_usage']}")
    
    # Baseline comparison (optional)
    if args.compare_baseline:
        print("\nBaseline comparison:")
        from transformers import AutoModelForCausalLM
        
        baseline = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            torch_dtype=dtype,
            device_map="auto",
            trust_remote_code=True
        )
        
        baseline_ppl = compute_perplexity(baseline, tokenizer, texts[:100], args.max_length, device)
        print(f"  Baseline: {baseline_ppl:.2f}")
        print(f"  SRDE: {srde_ppl:.2f}")
        print(f"  Improvement: {((baseline_ppl - srde_ppl) / baseline_ppl * 100):.1f}%")
        
        results["baseline_perplexity"] = baseline_ppl
        results["improvement_percent"] = (baseline_ppl - srde_ppl) / baseline_ppl * 100
    
    # Save results
    with open(args.output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.output_file}")
    
    print("\n" + "=" * 60)
    print("Evaluation Complete")
    print("=" * 60)


if __name__ == "__main__":
    main()
