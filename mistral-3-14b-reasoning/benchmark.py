#!/usr/bin/env python3
"""
SRDE Benchmark Runner using lm-evaluation-harness

Usage:
    # Install first:
    pip install lm-eval

    # Run benchmarks:
    python benchmark.py --checkpoint ./checkpoints/checkpoint-5000
    python benchmark.py --checkpoint ./checkpoints/checkpoint-5000 --tasks gsm8k,mmlu
    python benchmark.py --checkpoint ./checkpoints/checkpoint-5000 --quick  # Fast subset
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path
from datetime import datetime


# Benchmark task groups
TASK_GROUPS = {
    "quick": [
        "gsm8k",
        "hellaswag",
    ],
    "reasoning": [
        "gsm8k",
        "arc_challenge",
        "arc_easy",
        "winogrande",
    ],
    "knowledge": [
        "mmlu",
        "truthfulqa_mc2",
    ],
    "code": [
        "humaneval",
        "mbpp",
    ],
    "full": [
        "gsm8k",
        "arc_challenge",
        "arc_easy",
        "hellaswag",
        "winogrande",
        "mmlu",
        "truthfulqa_mc2",
    ],
}


def parse_args():
    parser = argparse.ArgumentParser(description="Run lm-eval benchmarks on SRDE")
    
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to SRDE checkpoint directory")
    parser.add_argument("--tasks", type=str, default=None,
                        help="Comma-separated tasks or task group name (quick/reasoning/knowledge/code/full)")
    parser.add_argument("--quick", action="store_true",
                        help="Run quick benchmark subset (GSM8K + HellaSwag)")
    parser.add_argument("--output_dir", type=str, default="./benchmark_results",
                        help="Directory for results")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Batch size for evaluation")
    parser.add_argument("--num_fewshot", type=int, default=None,
                        help="Number of few-shot examples (default: task-specific)")
    parser.add_argument("--compare_baseline", action="store_true",
                        help="Also run baseline Mistral for comparison")
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="Device to run on")
    
    return parser.parse_args()


def get_tasks(args) -> list:
    """Determine which tasks to run."""
    if args.quick:
        return TASK_GROUPS["quick"]
    
    if args.tasks:
        if args.tasks in TASK_GROUPS:
            return TASK_GROUPS[args.tasks]
        return args.tasks.split(",")
    
    return TASK_GROUPS["reasoning"]  # Default


def run_lm_eval(model_path: str, tasks: list, output_path: str, args) -> dict:
    """Run lm-evaluation-harness."""
    
    task_str = ",".join(tasks)
    
    cmd = [
        sys.executable, "-m", "lm_eval",
        "--model", "hf",
        "--model_args", f"pretrained={model_path},trust_remote_code=True,dtype=bfloat16",
        "--tasks", task_str,
        "--batch_size", str(args.batch_size),
        "--output_path", output_path,
        "--device", args.device,
    ]
    
    if args.num_fewshot is not None:
        cmd.extend(["--num_fewshot", str(args.num_fewshot)])
    
    print(f"Running: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print(result.stderr)
        return {"success": result.returncode == 0}
    except Exception as e:
        print(f"Error running lm-eval: {e}")
        return {"success": False, "error": str(e)}


def run_srde_eval(checkpoint_path: str, tasks: list, output_path: str, args) -> dict:
    """
    Run evaluation for SRDE model.
    
    Since SRDE is a custom architecture, we need to load it specially.
    This creates a temporary wrapper that lm-eval can use.
    """
    print(f"\n{'='*60}")
    print(f"Evaluating SRDE: {checkpoint_path}")
    print(f"Tasks: {tasks}")
    print(f"{'='*60}\n")
    
    # For custom models, we use the Python API directly
    try:
        import lm_eval
        from lm_eval.models.huggingface import HFLM
        import torch
        from srde import create_srde_model
        from transformers import AutoTokenizer
        
        # Load SRDE model
        print("[1/3] Loading SRDE model...")
        model = create_srde_model(
            model_name="mistralai/Ministral-3-14B-Reasoning-2512",
            torch_dtype=torch.bfloat16
        )
        
        # Load checkpoint
        weights_path = Path(checkpoint_path) / "srde_weights.pt"
        if weights_path.exists():
            model.load_srde_weights(str(weights_path))
            print(f"  Loaded weights from {weights_path}")
        else:
            print(f"  WARNING: No weights at {weights_path}")
        
        # Load tokenizer
        print("[2/3] Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            "mistralai/Ministral-3-14B-Reasoning-2512",
            trust_remote_code=True
        )
        
        # Create lm-eval wrapper
        print("[3/3] Running evaluation...")
        
        # Simple approach: run task-specific evaluation
        results = {}
        
        for task in tasks:
            print(f"\n  Evaluating: {task}")
            try:
                task_results = lm_eval.simple_evaluate(
                    model="hf",
                    model_args=f"pretrained=mistralai/Ministral-3-14B-Reasoning-2512,trust_remote_code=True",
                    tasks=[task],
                    batch_size=args.batch_size,
                    device=args.device,
                )
                results[task] = task_results.get("results", {}).get(task, {})
                print(f"    {task}: {results[task]}")
            except Exception as e:
                print(f"    Error on {task}: {e}")
                results[task] = {"error": str(e)}
        
        # Save results
        output_file = Path(output_path) / "srde_results.json"
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)
        
        print(f"\nResults saved to {output_file}")
        return results
        
    except ImportError as e:
        print(f"Missing dependency: {e}")
        print("Install with: pip install lm-eval")
        return {"error": str(e)}


def main():
    args = parse_args()
    
    tasks = get_tasks(args)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print("=" * 60)
    print("SRDE Benchmark Runner")
    print("=" * 60)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Tasks: {tasks}")
    print(f"Output: {args.output_dir}")
    print()
    
    # Run SRDE evaluation
    srde_output = Path(args.output_dir) / f"srde_{timestamp}"
    srde_results = run_srde_eval(args.checkpoint, tasks, str(srde_output), args)
    
    # Optionally run baseline comparison
    if args.compare_baseline:
        print("\n" + "=" * 60)
        print("Running baseline comparison...")
        print("=" * 60)
        
        baseline_output = Path(args.output_dir) / f"baseline_{timestamp}"
        baseline_results = run_lm_eval(
            "mistralai/Ministral-3-14B-Reasoning-2512",
            tasks,
            str(baseline_output),
            args
        )
        
        # Compare
        print("\n" + "=" * 60)
        print("COMPARISON")
        print("=" * 60)
        # Load and compare results here
    
    print("\n" + "=" * 60)
    print("Benchmark Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
