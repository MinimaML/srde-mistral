#!/usr/bin/env python3
"""
Upload SaRDinE model to HuggingFace Hub.

Usage:
    python upload_to_hf.py --token YOUR_HF_TOKEN --checkpoint ./checkpoints/checkpoint-5000
    python upload_to_hf.py --token YOUR_HF_TOKEN --repo MinimaML/SaRDinE-14B
"""
import argparse
import os
import json
from pathlib import Path
from huggingface_hub import HfApi, create_repo, upload_folder


def parse_args():
    parser = argparse.ArgumentParser(description="Upload SaRDinE to HuggingFace")
    parser.add_argument("--token", type=str, required=True, help="HuggingFace API token")
    parser.add_argument("--checkpoint", type=str, default="./checkpoints/checkpoint-latest", 
                        help="Path to checkpoint directory")
    parser.add_argument("--repo", type=str, default="MinimaML/SaRDinE-14B",
                        help="HuggingFace repo name (org/model)")
    parser.add_argument("--private", action="store_true", help="Make repo private")
    return parser.parse_args()


MODEL_CARD = """---
license: apache-2.0
language:
- en
tags:
- srde
- sparse-experts
- moe
- reasoning
base_model: mistralai/Ministral-3-14B-Reasoning-2512
library_name: transformers
---

# SaRDinE-14B

**S**parse **R**outed **D**elta **E**xperts on Mistral-14B-Reasoning.

## Model Description

SaRDinE is a novel Mixture-of-Experts architecture that augments a frozen base model with sparse delta experts. 
Unlike traditional MoE which fragments model capacity, SaRDinE uses 100% of the base model plus specialized expert deltas.

## Architecture Highlights

- **Base Model**: Mistral-14B-Reasoning (frozen)
- **Trainable Parameters**: ~2.4B (sparse deltas)
- **Experts**: 8 per layer, top-2 routing
- **Sparsity**: 1% per expert delta

## Usage

```python
# Requires the sardine package
# pip install sardine

from sardine import SaRDinE

model = SaRDinE.from_pretrained("MinimaML/SaRDinE-14B")
output = model.generate("Solve x^2 + 5x + 6 = 0")
```

## Training

Trained on domain-specific reasoning data using:
- Supervised expert pre-training (Phase 1)
- Joint fine-tuning with progressive unlock (Phase 2)

## Citation

If you use this model, please cite:
```
@misc{sardine2025,
  title={SaRDinE: Sparse Routed Delta Experts},
  author={MinimaML},
  year={2025},
  url={https://github.com/MinimaML/sardine}
}
```
"""


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


def main():
    args = parse_args()
    
    # Auto-detect checkpoint if not specified or default doesn't exist
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists() or args.checkpoint == "./checkpoints/checkpoint-latest":
        latest = find_latest_checkpoint("./checkpoints")
        if latest:
            print(f"Auto-detected latest checkpoint: {latest}")
            checkpoint_path = Path(latest)
        else:
            print("Error: No checkpoints found in ./checkpoints/")
            return
    
    if not checkpoint_path.exists():
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        return
    
    print(f"Uploading checkpoint: {checkpoint_path}")
    print(f"To repo: {args.repo}")
    
    # Initialize API
    api = HfApi(token=args.token)
    
    # Create repo if it doesn't exist
    try:
        create_repo(
            repo_id=args.repo,
            token=args.token,
            private=args.private,
            exist_ok=True
        )
        print(f"Repository {args.repo} ready")
    except Exception as e:
        print(f"Warning creating repo: {e}")
    
    # Create temp directory with required files
    upload_dir = checkpoint_path / "hf_upload"
    upload_dir.mkdir(exist_ok=True)
    
    # Copy/link SRDE weights
    import shutil
    srde_weights = checkpoint_path / "srde_weights.pt"
    if srde_weights.exists():
        shutil.copy(srde_weights, upload_dir / "srde_weights.pt")
    
    # Create model card
    with open(upload_dir / "README.md", "w") as f:
        f.write(MODEL_CARD)
    
    # Create config
    config = {
        "model_type": "srde",
        "base_model": "mistralai/Ministral-3-14B-Reasoning-2512",
        "num_experts": 8,
        "top_k": 2,
        "target_sparsity": 0.01,
        "architectures": ["SaRDinEForCausalLM"]
    }
    with open(upload_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    # Copy source files for trust_remote_code
    src_dir = Path(__file__).parent
    for src_file in ["srde.py", "config.py"]:
        src_path = src_dir / src_file
        if src_path.exists():
            shutil.copy(src_path, upload_dir / src_file)
    
    # Upload
    print("Uploading to HuggingFace...")
    upload_folder(
        folder_path=str(upload_dir),
        repo_id=args.repo,
        token=args.token,
        commit_message="Upload SaRDinE model"
    )
    
    print(f"\n✅ Upload complete!")
    print(f"View at: https://huggingface.co/{args.repo}")


if __name__ == "__main__":
    main()
