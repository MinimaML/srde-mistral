#!/usr/bin/env python3
"""
Upload SaRDinE model to HuggingFace Hub with trust_remote_code support.

Usage:
    python upload_to_hf.py --token YOUR_HF_TOKEN
    python upload_to_hf.py --token YOUR_HF_TOKEN --repo MinimaML/SaRDinE-14B8x1P
"""
import argparse
import os
import json
import shutil
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Upload SaRDinE to HuggingFace")
    parser.add_argument("--token", type=str, required=True, help="HuggingFace API token")
    parser.add_argument("--checkpoint", type=str, default=None, 
                        help="Path to checkpoint directory (auto-detects if not specified)")
    parser.add_argument("--repo", type=str, default="MinimaML/SaRDinE-14B8x4P",
                        help="HuggingFace repo name (org/model)")
    parser.add_argument("--private", action="store_true", help="Make repo private")
    return parser.parse_args()


def find_latest_checkpoint(checkpoints_dir: str = "./checkpoints") -> str:
    """Find the most recent checkpoint."""
    ckpt_path = Path(checkpoints_dir)
    
    if not ckpt_path.exists():
        return None
    
    # Check for LATEST marker file
    latest_marker = ckpt_path / "LATEST"
    if latest_marker.exists():
        checkpoint_name = latest_marker.read_text().strip()
        checkpoint = ckpt_path / checkpoint_name
        if checkpoint.exists():
            return str(checkpoint)
    
    # Check for checkpoint-latest symlink
    latest_link = ckpt_path / "checkpoint-latest"
    if latest_link.exists():
        return str(latest_link.resolve())
    
    # Find highest numbered checkpoint
    checkpoints = list(ckpt_path.glob("checkpoint-*"))
    if not checkpoints:
        return None
    
    def get_step(p):
        try:
            return int(p.name.split("-")[1])
        except:
            return -1
    
    checkpoints.sort(key=get_step, reverse=True)
    
    for ckpt in checkpoints:
        if (ckpt / "srde_weights.pt").exists():
            return str(ckpt)
    
    return None


MODEL_CARD = '''---
license: apache-2.0
language:
- en
tags:
- srde
- sparse-experts
- moe
- reasoning
- mistral
base_model: mistralai/Ministral-3-14B-Reasoning-2512
library_name: transformers
pipeline_tag: text-generation
datasets:
- openai/gsm8k
- meta-math/MetaMathQA
- google-research-datasets/mbpp
- allenai/sciq
- allenai/ai2_arc
---

# SaRDinE-14B8x4P

**S**parse **R**outed **D**elta **E**xperts on Mistral-14B-Reasoning.

> 14B base params, 8 experts per layer, ~4% sparsity (alpha)

## Model Description

SaRDinE is a novel Mixture-of-Experts architecture that augments a frozen base model with sparse delta experts. 
Unlike traditional MoE which fragments model capacity, SaRDinE uses 100% of the base model **plus** specialized expert deltas.

## Architecture

| Component | Value |
|-----------|-------|
| Base Model | Mistral-14B-Reasoning (frozen) |
| Trainable Parameters | ~2.4B (sparse deltas) |
| Experts | 8 per layer, top-2 routing |
| Sparsity | ~4% per expert delta (training) |
| Total Layers | 40 augmented |

## Usage

```python
import torch
from transformers import AutoTokenizer

# Load the SRDE model
from modeling_sardine import SaRDinEForCausalLM

model = SaRDinEForCausalLM.from_pretrained(
    "MinimaML/SaRDinE-14B8x1P",
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
)

tokenizer = AutoTokenizer.from_pretrained(
    "mistralai/Ministral-3-14B-Reasoning-2512",
    trust_remote_code=True
)

# Generate
prompt = "Solve: What is 15% of 80?"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=256)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

## Training

Trained on domain-specific reasoning data:
- **Phase 1**: Supervised expert pre-training (domain-specific)
- **Phase 2**: Joint fine-tuning with progressive expert unlocking

## Domains

| Domain | Expert Focus |
|--------|--------------|
| Math | GSM8K, MetaMathQA, Orca-Math |
| Logic | BigBench, CommonsenseQA |
| Code | MBPP, HumanEval, CodeFeedback |
| Science | SciQ, ARC, MMLU |
| Planning | HotpotQA, SCROLLS |
| Abstract | BigBench, AQuA-RAT |

## Citation

```bibtex
@misc{sardine2025,
  title={SaRDinE: Sparse Routed Delta Experts},
  author={MinimaML},
  year={2025},
  url={https://github.com/MinimaML/srde-mistral}
}
```

## License

Apache 2.0
'''


def main():
    args = parse_args()
    
    # Auto-detect checkpoint
    if args.checkpoint:
        checkpoint_path = Path(args.checkpoint)
    else:
        latest = find_latest_checkpoint("./checkpoints")
        if latest:
            checkpoint_path = Path(latest)
            print(f"Auto-detected checkpoint: {checkpoint_path}")
        else:
            print("Error: No checkpoints found!")
            return
    
    if not checkpoint_path.exists():
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        return
    
    print(f"Uploading checkpoint: {checkpoint_path}")
    print(f"To repo: {args.repo}")
    
    # Import here to fail fast if not installed
    from huggingface_hub import HfApi, create_repo, upload_folder
    
    # Create upload directory
    upload_dir = checkpoint_path / "hf_upload"
    if upload_dir.exists():
        shutil.rmtree(upload_dir)
    upload_dir.mkdir(parents=True)
    
    print("\n[1/5] Copying model weights...")
    srde_weights = checkpoint_path / "srde_weights.pt"
    if srde_weights.exists():
        shutil.copy(srde_weights, upload_dir / "srde_weights.pt")
        print(f"  Copied {srde_weights.stat().st_size / 1e9:.2f} GB")
    else:
        print("  ERROR: srde_weights.pt not found!")
        return
    
    print("\n[2/5] Creating model card...")
    with open(upload_dir / "README.md", "w") as f:
        f.write(MODEL_CARD)
    
    print("\n[3/5] Copying model code for trust_remote_code...")
    src_dir = Path(__file__).parent
    
    # Copy main model code
    for src_name, dst_name in [
        ("srde.py", "modeling_sardine.py"),
        ("config.py", "configuration_sardine.py"),
    ]:
        src_file = src_dir / src_name
        if src_file.exists():
            shutil.copy(src_file, upload_dir / dst_name)
            print(f"  {src_name} -> {dst_name}")
    
    # Create config.json for auto-loading
    config_json = {
        "model_type": "sardine",
        "architectures": ["SaRDinEForCausalLM"],
        "auto_map": {
            "AutoConfig": "configuration_sardine.SRDEConfig",
            "AutoModel": "modeling_sardine.SaRDinEForCausalLM",
            "AutoModelForCausalLM": "modeling_sardine.SaRDinEForCausalLM"
        },
        "base_model": "mistralai/Ministral-3-14B-Reasoning-2512",
        "num_experts": 8,
        "top_k": 2,
        "target_sparsity": 0.04,
        "torch_dtype": "bfloat16"
    }
    with open(upload_dir / "config.json", "w") as f:
        json.dump(config_json, f, indent=2)
    print("  Created config.json with auto_map")
    
    print("\n[4/5] Creating HuggingFace repo...")
    try:
        create_repo(
            repo_id=args.repo,
            token=args.token,
            private=args.private,
            exist_ok=True
        )
        print(f"  Repository ready: {args.repo}")
    except Exception as e:
        print(f"  Warning: {e}")
    
    print("\n[5/5] Uploading to HuggingFace...")
    upload_folder(
        folder_path=str(upload_dir),
        repo_id=args.repo,
        token=args.token,
        commit_message="Upload SaRDinE model with trust_remote_code support"
    )
    
    print(f"\n{'='*60}")
    print("✅ Upload complete!")
    print(f"{'='*60}")
    print(f"\nView at: https://huggingface.co/{args.repo}")
    print(f"\nTo use:")
    print(f"  from transformers import AutoModelForCausalLM")
    print(f'  model = AutoModelForCausalLM.from_pretrained("{args.repo}", trust_remote_code=True)')


if __name__ == "__main__":
    main()
