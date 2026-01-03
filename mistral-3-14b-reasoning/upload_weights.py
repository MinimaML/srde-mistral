
import os
import torch
import argparse
from pathlib import Path
from huggingface_hub import HfApi, login

from config import SRDEConfig
from srde import create_srde_model

def main():
    parser = argparse.ArgumentParser(description="Upload SRDE Checkpoints to HuggingFace")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint directory (e.g., checkpoints/final-29)")
    parser.add_argument("--repo_id", type=str, default="MinimaML/SaRDinE-14B6x1P", help="HF Repo ID")
    parser.add_argument("--base_model", type=str, default="mistralai/Ministral-3-14B-Reasoning-2512", help="Base model ID")
    args = parser.parse_args()

    if os.environ.get('HF_TOKEN'):
        login(token=os.environ['HF_TOKEN'])

    print(f"🚀 Preparing to upload {args.checkpoint} to {args.repo_id}...")

    # Load SRDE Model
    print("Loading model structure...")
    model = create_srde_model(args.base_model, torch_dtype=torch.bfloat16, device_map='cpu', trust_remote_code=True)
    
    weights_path = Path(args.checkpoint) / 'weights.pt'
    if not weights_path.exists():
        print(f"❌ Error: weights.pt not found in {args.checkpoint}")
        return

    print(f"Loading weights from {weights_path}...")
    try:
        model.load_srde_weights(str(weights_path))
        print("✅ Weights loaded successfully.")
    except Exception as e:
        print(f"❌ Failed to load weights: {e}")
        return

    # Create Repo
    api = HfApi()
    try:
        api.create_repo(repo_id=args.repo_id, exist_ok=True, private=True)
        print(f"Repo {args.repo_id} ready.")
    except Exception as e:
        print(f"Warning creating repo: {e}")

    # Save and Upload
    # Since SRDE is a custom architecture, we upload the raw weights.pt and config
    # The user can load it using the same srde.py codebase.
    # Future: We could merge into a full model, but for now, uploading the delta is efficient.
    
    print("Uploading weights and code...")
    try:
        api.upload_folder(
            folder_path=args.checkpoint,
            repo_id=args.repo_id,
            path_in_repo=f"checkpoints/{Path(args.checkpoint).name}",
            allow_patterns=["*.pt", "*.json", "*.md"]
        )
        
        # Also upload the code needed to run it
        api.upload_file(
            path_or_fileobj="srde.py",
            path_in_repo="srde.py",
            repo_id=args.repo_id
        )
        api.upload_file(
            path_or_fileobj="config.py",
            path_in_repo="config.py",
            repo_id=args.repo_id
        )
        
        print(f"✅ Successfully uploaded to https://huggingface.co/{args.repo_id}")
        
    except Exception as e:
        print(f"❌ Upload failed: {e}")

if __name__ == "__main__":
    main()
