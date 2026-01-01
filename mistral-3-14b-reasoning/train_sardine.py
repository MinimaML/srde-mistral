#!/usr/bin/env python3
"""
SaRDinE Training Script - Ready to Run

This script trains SRDE (Sparse Routed Delta Experts) on Mistral-14B-Reasoning.
Includes W&B logging and auto-upload to HuggingFace.

Usage:
    # Basic training
    python train_sardine.py --wandb_project sardine-training

    # With HF auto-upload
    python train_sardine.py --wandb_project sardine-training --hf_token YOUR_TOKEN

    # Multi-GPU
    torchrun --nproc_per_node=8 train_sardine.py --wandb_project sardine-training
"""

import os
import sys
import argparse
import json
import signal
import torch
import torch.nn as nn
from pathlib import Path
from datetime import datetime

# ============================================================================
# CONFIGURATION
# ============================================================================

DEFAULT_CONFIG = {
    # Model
    "model_name": "mistralai/Ministral-3-14B-Reasoning-2512",
    "num_experts": 8,
    "top_k": 2,
    "initial_sparsity": 0.05,
    "target_sparsity": 0.01,
    
    # Training
    "batch_size": 1,
    "gradient_accumulation_steps": 16,
    "learning_rate": 1e-4,
    "num_epochs": 3,
    "max_steps": 100000,
    "warmup_steps": 1000,
    
    # Data
    "max_length": 2048,
    "dataset": "MinimaML/srde-reasoning-mix",  # HuggingFace dataset
    
    # Logging & Checkpoints
    "log_steps": 10,
    "save_steps": 2500,
    "output_dir": "./checkpoints",
    
    # Hardware
    "bf16": True,
    "gradient_checkpointing": True,
}


def parse_args():
    parser = argparse.ArgumentParser(description="Train SaRDinE")
    
    # Required
    parser.add_argument("--wandb_project", type=str, required=True,
                        help="W&B project name")
    
    # Optional overrides
    parser.add_argument("--hf_token", type=str, default=None,
                        help="HuggingFace token for auto-upload")
    parser.add_argument("--hf_repo", type=str, default="MinimaML/SaRDinE-14B8x4P",
                        help="HuggingFace repo for upload")
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--max_steps", type=int, default=100000)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--save_steps", type=int, default=2500)
    parser.add_argument("--output_dir", type=str, default="./checkpoints")
    parser.add_argument("--resume_from", type=str, default=None)
    
    # Ablation flags
    parser.add_argument("--ablation_name", type=str, default=None,
                        help="Name for ablation study (e.g., 'no_router', '4_experts')")
    parser.add_argument("--num_experts", type=int, default=8)
    parser.add_argument("--target_sparsity", type=float, default=0.01)
    
    return parser.parse_args()


# ============================================================================
# SETUP
# ============================================================================

def setup_wandb(args):
    """Initialize Weights & Biases logging."""
    import wandb
    
    run_name = f"sardine-{datetime.now().strftime('%Y%m%d-%H%M')}"
    if args.ablation_name:
        run_name = f"ablation-{args.ablation_name}-{run_name}"
    
    wandb.init(
        project=args.wandb_project,
        name=run_name,
        config={
            **DEFAULT_CONFIG,
            "num_experts": args.num_experts,
            "target_sparsity": args.target_sparsity,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "ablation": args.ablation_name,
        }
    )
    return wandb


def setup_model(args):
    """Load base model and create SRDE wrapper."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    print(f"Loading base model: {DEFAULT_CONFIG['model_name']}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        DEFAULT_CONFIG["model_name"],
        trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load base model
    dtype = torch.bfloat16 if DEFAULT_CONFIG["bf16"] else torch.float32
    base_model = AutoModelForCausalLM.from_pretrained(
        DEFAULT_CONFIG["model_name"],
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=True
    )
    
    # Import and wrap with SRDE
    # Note: This assumes srde.py and config.py are in the same directory
    from srde import create_srde_model
    from config import SRDEConfig
    
    srde_config = SRDEConfig(
        num_experts=args.num_experts,
        top_k=DEFAULT_CONFIG["top_k"],
        initial_sparsity=DEFAULT_CONFIG["initial_sparsity"],
        target_sparsity=args.target_sparsity,
    )
    
    model = create_srde_model(
        model_name=DEFAULT_CONFIG["model_name"],
        srde_config=srde_config,
        torch_dtype=dtype
    )
    
    if DEFAULT_CONFIG["gradient_checkpointing"]:
        model.gradient_checkpointing_enable()
    
    print(f"SRDE model created with {args.num_experts} experts")
    return model, tokenizer


def setup_data(tokenizer, args):
    """Load and prepare training data."""
    from datasets import load_dataset
    from torch.utils.data import DataLoader
    
    print(f"Loading dataset: {DEFAULT_CONFIG['dataset']}")
    
    try:
        dataset = load_dataset(DEFAULT_CONFIG["dataset"], split="train")
    except:
        print("Dataset not found, using placeholder")
        # Create minimal placeholder
        dataset = None
        return None
    
    def tokenize_fn(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=DEFAULT_CONFIG["max_length"],
            padding="max_length",
            return_tensors="pt"
        )
    
    dataset = dataset.map(tokenize_fn, batched=True, remove_columns=dataset.column_names)
    dataset.set_format("torch")
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    return dataloader


def setup_optimizer(model, args):
    """Create optimizer with separate param groups."""
    # SRDE params only (base model is frozen)
    srde_params = [p for n, p in model.named_parameters() if p.requires_grad]
    
    optimizer = torch.optim.AdamW(
        srde_params,
        lr=args.learning_rate,
        weight_decay=0.01
    )
    
    print(f"Optimizer created for {len(srde_params)} parameter groups")
    return optimizer


# ============================================================================
# TRAINING LOOP
# ============================================================================

def train(model, tokenizer, dataloader, optimizer, args, wandb):
    """Main training loop with W&B logging."""
    
    device = next(model.parameters()).device
    scaler = torch.cuda.amp.GradScaler() if DEFAULT_CONFIG["bf16"] else None
    
    global_step = 0
    best_loss = float('inf')
    
    # Resume from checkpoint
    if args.resume_from:
        checkpoint = torch.load(args.resume_from)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        global_step = checkpoint.get("step", 0)
        print(f"Resumed from step {global_step}")
    
    print(f"\n{'='*60}")
    print("Starting Training")
    print(f"{'='*60}")
    
    model.train()
    
    for epoch in range(args.num_epochs):
        epoch_loss = 0
        epoch_steps = 0
        
        for batch_idx, batch in enumerate(dataloader):
            if global_step >= args.max_steps:
                break
            
            # Move to device
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            
            # Forward pass
            with torch.cuda.amp.autocast(enabled=DEFAULT_CONFIG["bf16"]):
                outputs = model(input_ids, attention_mask=attention_mask, labels=input_ids)
                loss = outputs.loss / args.gradient_accumulation_steps
            
            # Backward pass
            if scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # Optimizer step
            if (batch_idx + 1) % args.gradient_accumulation_steps == 0:
                if scaler:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad()
                global_step += 1
                
                # Logging
                if global_step % DEFAULT_CONFIG["log_steps"] == 0:
                    current_loss = loss.item() * args.gradient_accumulation_steps
                    wandb.log({
                        "loss": current_loss,
                        "step": global_step,
                        "epoch": epoch,
                        "lr": optimizer.param_groups[0]["lr"],
                    })
                    print(f"Step {global_step} | Loss: {current_loss:.4f}")
                
                # Save checkpoint
                if global_step % args.save_steps == 0:
                    save_checkpoint(model, optimizer, global_step, args)
                    
                    if current_loss < best_loss:
                        best_loss = current_loss
                        save_checkpoint(model, optimizer, global_step, args, is_best=True)
                
                epoch_loss += loss.item() * args.gradient_accumulation_steps
                epoch_steps += 1
        
        # End of epoch
        avg_epoch_loss = epoch_loss / max(1, epoch_steps)
        print(f"\nEpoch {epoch} complete | Avg Loss: {avg_epoch_loss:.4f}")
        wandb.log({"epoch_loss": avg_epoch_loss, "epoch": epoch})
    
    # Final save
    save_checkpoint(model, optimizer, global_step, args, is_final=True)
    
    # Auto-upload to HuggingFace
    if args.hf_token:
        upload_to_hf(args)
    
    print(f"\n{'='*60}")
    print("Training Complete!")
    print(f"{'='*60}")
    
    return global_step


def save_checkpoint(model, optimizer, step, args, is_best=False, is_final=False):
    """Save training checkpoint."""
    output_dir = Path(args.output_dir)
    
    if is_best:
        save_path = output_dir / "checkpoint-best"
    elif is_final:
        save_path = output_dir / "checkpoint-final"
    else:
        save_path = output_dir / f"checkpoint-{step}"
    
    save_path.mkdir(parents=True, exist_ok=True)
    
    # Save SRDE weights only
    model.save_srde_weights(str(save_path / "srde_weights.pt"))
    
    # Save training state
    torch.save({
        "optimizer": optimizer.state_dict(),
        "step": step,
    }, save_path / "training_state.pt")
    
    print(f"Saved checkpoint to {save_path}")


def upload_to_hf(args):
    """Upload trained model to HuggingFace."""
    from huggingface_hub import HfApi, upload_folder
    
    print("\nUploading to HuggingFace...")
    
    api = HfApi()
    
    # Upload final checkpoint
    final_path = Path(args.output_dir) / "checkpoint-final"
    if final_path.exists():
        api.upload_folder(
            folder_path=str(final_path),
            repo_id=args.hf_repo,
            token=args.hf_token,
            commit_message=f"Training complete - uploaded automatically"
        )
        print(f"Uploaded to {args.hf_repo}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    args = parse_args()
    
    # Install dependencies if needed
    try:
        import wandb
    except ImportError:
        os.system("pip install wandb")
        import wandb
    
    # Setup
    wandb_run = setup_wandb(args)
    model, tokenizer = setup_model(args)
    dataloader = setup_data(tokenizer, args)
    optimizer = setup_optimizer(model, args)
    
    if dataloader is None:
        print("ERROR: Could not load dataset. Please check dataset availability.")
        return
    
    # Signal handler for graceful shutdown
    def signal_handler(sig, frame):
        print("\nReceived interrupt, saving checkpoint...")
        save_checkpoint(model, optimizer, global_step, args)
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Train
    final_step = train(model, tokenizer, dataloader, optimizer, args, wandb_run)
    
    # Finish W&B
    wandb_run.finish()
    
    print(f"\nTraining finished at step {final_step}")
    print(f"Checkpoints saved to: {args.output_dir}")
    if args.hf_token:
        print(f"Model uploaded to: {args.hf_repo}")


if __name__ == "__main__":
    main()
