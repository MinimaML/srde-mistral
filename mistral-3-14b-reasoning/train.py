"""
SRDE Training Script for Mistral-3-14B-Reasoning

ROBUST VERSION with:
- Automatic checkpoint resumption
- Crash recovery
- Gradient checkpointing
- Mixed precision training
- Progress persistence
- Heartbeat logging
"""
import os
import sys
import argparse
import json
import signal
import time
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    get_linear_schedule_with_warmup
)
from tqdm import tqdm

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

from config import SRDEConfig, MISTRAL_14B_SRDE_CONFIG
from srde import SRDEModel, create_srde_model
from losses import SRDELoss
from scheduler import SparsityScheduler, TemperatureScheduler, PhaseScheduler

# Optional Muon optimizer
try:
    from muon import MuonAdamW
    MUON_AVAILABLE = True
except ImportError:
    MUON_AVAILABLE = False


#===============================================================================
# GLOBAL STATE FOR CRASH RECOVERY
#===============================================================================

TRAINING_STATE = {
    "global_step": 0,
    "epoch": 0,
    "best_loss": float("inf"),
    "last_checkpoint": None,
    "start_time": None,
    "last_heartbeat": None
}

SHOULD_STOP = False


def signal_handler(signum, frame):
    """Handle interrupts gracefully."""
    global SHOULD_STOP
    print(f"\n[SIGNAL] Received signal {signum}, finishing current step...")
    SHOULD_STOP = True


signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


#===============================================================================
# ARGUMENT PARSING
#===============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Train SRDE on Mistral-3-14B (Robust)")
    
    # Model
    parser.add_argument("--model_name", type=str, default="mistralai/Ministral-3-14B-Reasoning-2512")
    parser.add_argument("--output_dir", type=str, default="./srde_checkpoints")
    parser.add_argument("--resume_from", type=str, default=None, help="Checkpoint to resume from")
    
    # Data
    parser.add_argument("--train_file", type=str, default=None, help="JSONL training file")
    parser.add_argument("--pretokenized_dir", type=str, default=None, help="Pre-tokenized data directory")
    parser.add_argument("--max_length", type=int, default=2048)
    
    # Training
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--warmup_steps", type=int, default=100)
    
    # SRDE config
    parser.add_argument("--num_experts", type=int, default=8)
    parser.add_argument("--top_k", type=int, default=2)
    parser.add_argument("--initial_sparsity", type=float, default=0.05)
    parser.add_argument("--target_sparsity", type=float, default=0.01)
    
    # Logging & Checkpointing
    parser.add_argument("--log_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--heartbeat_steps", type=int, default=50)
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="srde-mistral")
    parser.add_argument("--wandb_run_name", type=str, default=None)
    
    # Hardware & Optimization
    parser.add_argument("--bf16", action="store_true", default=True)
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--compile", action="store_true", help="Use torch.compile")
    parser.add_argument("--compile_mode", type=str, default="reduce-overhead", 
                       choices=["default", "reduce-overhead", "max-autotune"],
                       help="torch.compile mode")
    
    # Muon optimizer (faster training)
    parser.add_argument("--use_muon", action="store_true", help="Use Muon optimizer (~35%% faster)")
    parser.add_argument("--muon_lr", type=float, default=0.02, help="Muon learning rate")
    
    # Flash Attention 3 (H100/H200)
    parser.add_argument("--flash_attention", action="store_true", help="Use Flash Attention 3 (requires H100/H200)")
    
    # Supervised expert training (Phase 1: pre-train experts on domain data)
    parser.add_argument("--supervised_experts", action="store_true", 
                       help="Two-phase training: pre-train experts on domain-specific data, then train router")
    parser.add_argument("--expert_pretrain_steps", type=int, default=1000,
                       help="Steps to pre-train each expert on its domain data")
    parser.add_argument("--jsonl_data", type=str, default=None,
                       help="JSONL file with domain labels (required for --supervised_experts)")
    
    # Router warm-start
    parser.add_argument("--warm_start", action="store_true",
                       help="Initialize routers from hidden state clustering (K-means)")
    
    # Progressive expert unlocking
    parser.add_argument("--progressive_unlock", type=str, default=None,
                       choices=["linear", "warmup", "all"],
                       help="Progressive expert unlocking schedule (linear=gradual, warmup=1 then all)")
    
    return parser.parse_args()


#===============================================================================
# DATASET
#===============================================================================

class RobustDataset(Dataset):
    """JSONL dataset with error handling."""
    
    def __init__(self, file_path: str, tokenizer, max_length: int):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = []
        
        print(f"Loading dataset from {file_path}...")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if line.strip():
                    try:
                        self.data.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        print(f"Warning: Skipping malformed line {i}: {e}")
        
        print(f"Loaded {len(self.data)} examples")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        text = item.get("text", item.get("content", str(item)))
        
        try:
            encodings = self.tokenizer(
                text,
                truncation=True,
                max_length=self.max_length,
                padding="max_length",
                return_tensors="pt"
            )
            return {
                "input_ids": encodings["input_ids"].squeeze(0),
                "attention_mask": encodings["attention_mask"].squeeze(0),
                "labels": encodings["input_ids"].squeeze(0)
            }
        except Exception as e:
            print(f"Error tokenizing example {idx}: {e}")
            # Return a dummy example
            dummy = torch.zeros(self.max_length, dtype=torch.long)
            return {"input_ids": dummy, "attention_mask": dummy, "labels": dummy}


class DomainDataset(Dataset):
    """Dataset filtered by domain_id for supervised expert training."""
    
    def __init__(self, file_path: str, tokenizer, max_length: int, domain_id: int):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.domain_id = domain_id
        self.data = []
        
        print(f"Loading domain {domain_id} data from {file_path}...")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    try:
                        item = json.loads(line)
                        if item.get("domain_id") == domain_id:
                            self.data.append(item)
                    except json.JSONDecodeError:
                        continue
        
        print(f"  Loaded {len(self.data)} examples for domain {domain_id}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        text = item.get("text", "")
        
        try:
            encodings = self.tokenizer(
                text,
                truncation=True,
                max_length=self.max_length,
                padding="max_length",
                return_tensors="pt"
            )
            return {
                "input_ids": encodings["input_ids"].squeeze(0),
                "attention_mask": encodings["attention_mask"].squeeze(0),
                "labels": encodings["input_ids"].squeeze(0)
            }
        except Exception as e:
            dummy = torch.zeros(self.max_length, dtype=torch.long)
            return {"input_ids": dummy, "attention_mask": dummy, "labels": dummy}


#===============================================================================
# CHECKPOINTING
#===============================================================================

def save_checkpoint(
    model: SRDEModel,
    optimizer: torch.optim.Optimizer,
    lr_scheduler,
    schedulers: dict,
    step: int,
    epoch: int,
    loss: float,
    output_dir: str,
    is_best: bool = False
):
    """Save full training state for resume."""
    path = Path(output_dir) / f"checkpoint-{step}"
    path.mkdir(parents=True, exist_ok=True)
    
    # Save SRDE weights
    model.save_srde_weights(str(path / "srde_weights.pt"))
    
    # Save optimizer state
    torch.save(optimizer.state_dict(), path / "optimizer.pt")
    
    # Save LR scheduler
    torch.save(lr_scheduler.state_dict(), path / "lr_scheduler.pt")
    
    # Save SRDE schedulers
    scheduler_states = {name: sched.state_dict() for name, sched in schedulers.items()}
    torch.save(scheduler_states, path / "schedulers.pt")
    
    # Save training state
    state = {
        "step": step,
        "epoch": epoch,
        "loss": loss,
        "timestamp": datetime.now().isoformat(),
        "best_loss": TRAINING_STATE["best_loss"]
    }
    with open(path / "training_state.json", "w") as f:
        json.dump(state, f, indent=2)
    
    # Symlink to latest
    latest_link = Path(output_dir) / "checkpoint-latest"
    if latest_link.exists():
        latest_link.unlink()
    try:
        latest_link.symlink_to(path.name)
    except OSError:
        # Windows doesn't always support symlinks
        with open(Path(output_dir) / "LATEST", "w") as f:
            f.write(path.name)
    
    # Save best separately
    if is_best:
        best_path = Path(output_dir) / "checkpoint-best"
        if best_path.exists():
            import shutil
            shutil.rmtree(best_path)
        import shutil
        shutil.copytree(path, best_path)
    
    print(f"[CHECKPOINT] Saved to {path}" + (" (best)" if is_best else ""))
    return str(path)


def load_checkpoint(
    checkpoint_path: str,
    model: SRDEModel,
    optimizer: torch.optim.Optimizer,
    lr_scheduler,
    schedulers: dict
) -> dict:
    """Load training state from checkpoint."""
    path = Path(checkpoint_path)
    
    print(f"[RESUME] Loading checkpoint from {path}")
    
    # Load SRDE weights
    model.load_srde_weights(str(path / "srde_weights.pt"))
    
    # Load optimizer
    if (path / "optimizer.pt").exists():
        optimizer.load_state_dict(torch.load(path / "optimizer.pt", map_location="cpu"))
    
    # Load LR scheduler
    if (path / "lr_scheduler.pt").exists():
        lr_scheduler.load_state_dict(torch.load(path / "lr_scheduler.pt", map_location="cpu"))
    
    # Load SRDE schedulers
    if (path / "schedulers.pt").exists():
        sched_states = torch.load(path / "schedulers.pt", map_location="cpu")
        for name, sched in schedulers.items():
            if name in sched_states:
                sched.load_state_dict(sched_states[name])
    
    # Load training state
    state = {}
    if (path / "training_state.json").exists():
        with open(path / "training_state.json") as f:
            state = json.load(f)
    
    print(f"[RESUME] Restored step {state.get('step', 0)}, epoch {state.get('epoch', 0)}")
    return state


def find_latest_checkpoint(output_dir: str) -> Optional[str]:
    """Find the latest checkpoint to resume from."""
    output_path = Path(output_dir)
    
    # Check for explicit "latest" marker
    latest_marker = output_path / "LATEST"
    if latest_marker.exists():
        checkpoint_name = latest_marker.read_text().strip()
        checkpoint_path = output_path / checkpoint_name
        if checkpoint_path.exists():
            return str(checkpoint_path)
    
    # Or symlink
    latest_link = output_path / "checkpoint-latest"
    if latest_link.exists():
        return str(latest_link.resolve())
    
    # Find highest numbered checkpoint
    checkpoints = list(output_path.glob("checkpoint-*"))
    if not checkpoints:
        return None
    
    # Sort by step number
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


#===============================================================================
# SUPERVISED EXPERT PRE-TRAINING (Phase 1)
#===============================================================================

def pretrain_experts_supervised(
    model: SRDEModel,
    tokenizer,
    jsonl_path: str,
    num_domains: int,
    steps_per_expert: int,
    max_length: int,
    learning_rate: float = 1e-4,
    device: str = "cuda"
):
    """
    Phase 1: Pre-train each expert on its domain-specific data.
    
    This gives each expert a head start on its specialization before
    the router learns to select them.
    """
    print("\n" + "="*60)
    print("PHASE 1: Supervised Expert Pre-Training")
    print("="*60)
    
    # We'll train experts in each SRDE layer
    for layer_key, srde_layer in model.srde_layers.items():
        print(f"\n[Layer {layer_key}] Pre-training {len(srde_layer.experts)} experts...")
        
        for expert_idx in range(len(srde_layer.experts)):
            # Map expert to domain (expert 0 -> domain 0, etc.)
            domain_id = expert_idx % num_domains
            
            print(f"  Expert {expert_idx} <- Domain {domain_id}", end=" ", flush=True)
            
            # Create domain-specific dataset
            domain_dataset = DomainDataset(
                jsonl_path, tokenizer, max_length, domain_id
            )
            
            if len(domain_dataset) == 0:
                print("(no data, skipping)")
                continue
            
            domain_loader = DataLoader(
                domain_dataset, batch_size=1, shuffle=True, num_workers=0
            )
            
            # Get expert parameters
            expert = srde_layer.experts[expert_idx]
            expert_params = [p for p in expert.parameters() if p.requires_grad]
            
            if not expert_params:
                print("(no trainable params, skipping)")
                continue
            
            optimizer = AdamW(expert_params, lr=learning_rate)
            
            # Pre-train this expert
            expert.train()
            total_loss = 0.0
            step = 0
            
            for batch in domain_loader:
                if step >= steps_per_expert:
                    break
                
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)
                
                optimizer.zero_grad()
                
                # Forward through the full model but only update this expert
                with torch.no_grad():
                    outputs = model(input_ids, attention_mask=attention_mask)
                
                # Compute a simple reconstruction loss for the expert
                # (This is a simplified approach - expert learns to minimize task loss)
                logits = outputs.get('logits')
                if logits is not None:
                    loss = torch.nn.functional.cross_entropy(
                        logits[:, :-1].reshape(-1, logits.size(-1)),
                        labels[:, 1:].reshape(-1),
                        ignore_index=0
                    )
                    
                    # Only backprop through the expert
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
                
                step += 1
            
            avg_loss = total_loss / max(step, 1)
            print(f"(loss={avg_loss:.4f}, steps={step})")
    
    print("\n" + "="*60)
    print("Phase 1 Complete - All experts pre-trained on domain data")
    print("="*60 + "\n")


def warm_start_routers(
    model: SRDEModel,
    tokenizer,
    jsonl_path: str,
    num_samples: int = 1000,
    max_length: int = 512,
    device: str = "cuda"
):
    """
    Warm-start all router weights from hidden state clustering.
    
    Collects hidden states from sample data and uses K-means to 
    initialize router weights for better expert assignment from the start.
    """
    print("\n[WARM-START] Initializing routers from hidden state clustering...")
    
    # Collect sample hidden states
    hidden_samples = []
    
    with open(jsonl_path, 'r') as f:
        lines = [l for l in f if l.strip()][:num_samples]
    
    model.eval()
    with torch.no_grad():
        for line in lines[:min(100, len(lines))]:  # Limit for speed
            try:
                item = json.loads(line)
                text = item.get("text", "")
                
                encodings = tokenizer(
                    text,
                    truncation=True,
                    max_length=max_length,
                    return_tensors="pt"
                )
                
                input_ids = encodings["input_ids"].to(device)
                
                # Get hidden states from a forward pass
                outputs = model(input_ids, output_hidden_states=True)
                hidden = outputs.get('hidden_states')
                if hidden is not None and len(hidden) > 0:
                    # Use last hidden state, average across sequence
                    last_hidden = hidden[-1].mean(dim=1)  # [batch, hidden]
                    hidden_samples.append(last_hidden.cpu())
            except Exception as e:
                continue
    
    if not hidden_samples:
        print("[WARM-START] No hidden states collected, skipping")
        return
    
    hidden_states = torch.cat(hidden_samples, dim=0)
    print(f"[WARM-START] Collected {hidden_states.shape[0]} hidden state samples")
    
    # Warm-start each router
    for layer_key, srde_layer in model.srde_layers.items():
        srde_layer.router.warm_start_from_hidden_states(hidden_states, method="kmeans")
    
    print("[WARM-START] All routers initialized from K-means clustering")
    model.train()


def progressive_unlock_schedule(
    model: SRDEModel,
    current_step: int,
    total_steps: int,
    unlock_schedule: str = "linear"
) -> int:
    """
    Progressively unlock experts based on training progress.
    
    Returns the number of experts that should be unlocked at this step.
    """
    num_experts = model.config.num_experts
    
    if unlock_schedule == "linear":
        # Unlock one expert at a time, evenly spaced
        experts_per_chunk = total_steps // num_experts
        unlocked = min(num_experts, (current_step // experts_per_chunk) + 1)
    elif unlock_schedule == "warmup":
        # Start with 1 expert, unlock rest after 10% of training
        warmup_steps = total_steps // 10
        if current_step < warmup_steps:
            unlocked = 1
        else:
            unlocked = num_experts
    elif unlock_schedule == "all":
        # All experts from the start
        unlocked = num_experts
    else:
        unlocked = num_experts
    
    # Apply to all SRDE layers
    for layer_key, srde_layer in model.srde_layers.items():
        current_unlocked = srde_layer.get_unlocked_expert_count()
        if current_unlocked != unlocked:
            srde_layer.set_unlocked_experts(list(range(unlocked)))
    
    return unlocked


#===============================================================================
# TRAINING
#===============================================================================

def train_step(
    model: SRDEModel,
    batch: Dict[str, torch.Tensor],
    loss_fn: SRDELoss
) -> Dict[str, torch.Tensor]:
    """Single training step with error handling."""
    
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    labels = batch["labels"]
    
    # Forward pass
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels
    )
    
    # Get task loss
    if hasattr(outputs, 'loss'):
        task_loss = outputs.loss
    elif isinstance(outputs, dict) and 'loss' in outputs:
        task_loss = outputs['loss']
    else:
        logits = outputs['logits'] if isinstance(outputs, dict) else outputs.logits
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        task_loss = nn.functional.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100
        )
    
    # Auxiliary loss
    aux_loss = outputs.get('aux_loss', torch.tensor(0.0, device=task_loss.device))
    if not isinstance(aux_loss, torch.Tensor):
        aux_loss = torch.tensor(0.0, device=task_loss.device)
    
    total_loss = task_loss + aux_loss
    
    return {
        "loss": total_loss,
        "task_loss": task_loss.detach(),
        "aux_loss": aux_loss.detach() if isinstance(aux_loss, torch.Tensor) else aux_loss
    }


def write_heartbeat(output_dir: str, step: int, loss: float, phase: str):
    """Write heartbeat file for external monitoring."""
    heartbeat = {
        "step": step,
        "loss": loss,
        "phase": phase,
        "timestamp": datetime.now().isoformat(),
        "pid": os.getpid()
    }
    with open(Path(output_dir) / "heartbeat.json", "w") as f:
        json.dump(heartbeat, f)


def main():
    global SHOULD_STOP, TRAINING_STATE
    
    args = parse_args()
    
    # Setup output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Write startup marker
    with open(Path(args.output_dir) / "TRAINING_STARTED", "w") as f:
        f.write(datetime.now().isoformat())
    
    print("="*60)
    print("SRDE Training - Robust Version")
    print("="*60)
    
    #---------------------------------------------------------------------------
    # Initialize wandb
    #---------------------------------------------------------------------------
    if args.use_wandb and WANDB_AVAILABLE:
        run_name = args.wandb_run_name or f"srde-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        wandb.init(project=args.wandb_project, name=run_name, config=vars(args), resume="allow")
    
    #---------------------------------------------------------------------------
    # Load tokenizer
    #---------------------------------------------------------------------------
    print("\n[1/6] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    #---------------------------------------------------------------------------
    # Create SRDE config and model
    #---------------------------------------------------------------------------
    print("\n[2/6] Creating SRDE model...")
    config = SRDEConfig(
        num_experts=args.num_experts,
        top_k=args.top_k,
        initial_sparsity=args.initial_sparsity,
        target_sparsity=args.target_sparsity
    )
    
    dtype = torch.bfloat16 if args.bf16 else torch.float32
    model = create_srde_model(
        model_name=args.model_name,
        config=config,
        torch_dtype=dtype,
        use_flash_attention=args.flash_attention
    )
    
    if args.gradient_checkpointing:
        model.base_model.gradient_checkpointing_enable()
        print("  Gradient checkpointing enabled")
    
    if args.compile and hasattr(torch, 'compile'):
        print(f"  Compiling model with torch.compile (mode={args.compile_mode})...")
        model = torch.compile(model, mode=args.compile_mode)
    
    #---------------------------------------------------------------------------
    # Phase 1: Supervised Expert Pre-Training (optional)
    #---------------------------------------------------------------------------
    if args.supervised_experts:
        if not args.jsonl_data:
            # Try to find jsonl in work_dir
            potential_jsonl = Path(args.pretokenized_dir).parent / "data.jsonl" if args.pretokenized_dir else None
            if potential_jsonl and potential_jsonl.exists():
                args.jsonl_data = str(potential_jsonl)
            else:
                raise ValueError("--supervised_experts requires --jsonl_data with domain labels")
        
        print(f"\n[2.5/6] Phase 1: Supervised Expert Pre-Training...")
        pretrain_experts_supervised(
            model=model,
            tokenizer=tokenizer,
            jsonl_path=args.jsonl_data,
            num_domains=6,  # math, logic, code, science, planning, abstract
            steps_per_expert=args.expert_pretrain_steps,
            max_length=args.max_length,
            learning_rate=args.learning_rate,
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
    
    #---------------------------------------------------------------------------
    # Load dataset
    #---------------------------------------------------------------------------
    print(f"\n[3/6] Loading training data...")
    
    if args.pretokenized_dir:
        # Use pre-tokenized data (faster)
        from pretokenized_dataset import create_dataloader
        print(f"  Using pre-tokenized data from: {args.pretokenized_dir}")
        dataloader = create_dataloader(
            args.pretokenized_dir,
            batch_size=args.batch_size,
            max_length=args.max_length,
            num_workers=4,
            shuffle=True
        )
    elif args.train_file:
        # Use JSONL with on-the-fly tokenization
        dataset = RobustDataset(args.train_file, tokenizer, args.max_length)
        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True
        )
    else:
        raise ValueError("Must specify --train_file or --pretokenized_dir")
    
    #---------------------------------------------------------------------------
    # Setup optimizer and schedulers
    #---------------------------------------------------------------------------
    print("\n[4/6] Setting up optimizer and schedulers...")
    trainable_params = model.get_trainable_params()
    
    # Choose optimizer: Muon+AdamW (faster) or AdamW (standard)
    if args.use_muon and MUON_AVAILABLE:
        print("  Using Muon optimizer (faster training)")
        optimizer = MuonAdamW(
            model,
            muon_lr=args.muon_lr,
            adamw_lr=args.learning_rate,
            weight_decay=0.01
        )
        # Muon doesn't use LR scheduler the same way
        lr_scheduler_enabled = False
    elif args.use_muon and not MUON_AVAILABLE:
        print("  [WARN] Muon not available, falling back to AdamW")
        optimizer = AdamW(trainable_params, lr=args.learning_rate, weight_decay=0.01)
        lr_scheduler_enabled = True
    else:
        optimizer = AdamW(trainable_params, lr=args.learning_rate, weight_decay=0.01)
        lr_scheduler_enabled = True
    
    steps_per_epoch = len(dataloader) // args.gradient_accumulation_steps
    total_steps = steps_per_epoch * args.num_epochs
    if args.max_steps > 0:
        total_steps = min(total_steps, args.max_steps)
    
    if lr_scheduler_enabled:
        lr_scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=args.warmup_steps,
            num_training_steps=total_steps
        )
    else:
        # Dummy scheduler for Muon
        lr_scheduler = None
    
    sparsity_scheduler = SparsityScheduler(
        initial_sparsity=args.initial_sparsity,
        target_sparsity=args.target_sparsity,
        warmup_steps=total_steps // 2
    )
    
    temperature_scheduler = TemperatureScheduler(
        initial_temp=1.0,
        min_temp=0.1,
        anneal_steps=total_steps
    )
    
    phase_scheduler = PhaseScheduler(config)
    
    schedulers = {
        "sparsity": sparsity_scheduler,
        "temperature": temperature_scheduler,
        "phase": phase_scheduler
    }
    
    loss_fn = SRDELoss(config)
    
    #---------------------------------------------------------------------------
    # Resume from checkpoint if specified
    #---------------------------------------------------------------------------
    start_step = 0
    start_epoch = 0
    
    resume_path = args.resume_from
    if resume_path is None:
        # Auto-detect latest checkpoint
        resume_path = find_latest_checkpoint(args.output_dir)
    
    if resume_path:
        print(f"\n[5/6] Resuming from checkpoint: {resume_path}")
        state = load_checkpoint(resume_path, model, optimizer, lr_scheduler, schedulers)
        start_step = state.get("step", 0)
        start_epoch = state.get("epoch", 0)
        TRAINING_STATE["best_loss"] = state.get("best_loss", float("inf"))
    else:
        print("\n[5/6] Starting fresh training (no checkpoint found)")
    
    #---------------------------------------------------------------------------
    # Training loop
    #---------------------------------------------------------------------------
    print(f"\n[6/6] Starting training loop...")
    print(f"  Total steps: {total_steps}")
    print(f"  Starting from step: {start_step}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Gradient accumulation: {args.gradient_accumulation_steps}")
    print(f"  Effective batch size: {args.batch_size * args.gradient_accumulation_steps}")
    print(f"  Trainable params: {sum(p.numel() for p in trainable_params):,}")
    print("="*60)
    
    TRAINING_STATE["start_time"] = time.time()
    
    global_step = start_step
    accumulated_loss = 0.0
    num_accumulated = 0
    
    model.train()
    
    for epoch in range(start_epoch, args.num_epochs):
        TRAINING_STATE["epoch"] = epoch
        
        progress = tqdm(
            enumerate(dataloader),
            total=len(dataloader),
            desc=f"Epoch {epoch+1}",
            initial=0
        )
        
        for batch_idx, batch in progress:
            # Check for stop signal
            if SHOULD_STOP:
                print("\n[STOP] Graceful shutdown requested, saving checkpoint...")
                save_checkpoint(
                    model, optimizer, lr_scheduler, schedulers,
                    global_step, epoch, accumulated_loss / max(num_accumulated, 1),
                    args.output_dir
                )
                print("[STOP] Checkpoint saved, exiting.")
                sys.exit(0)
            
            # Skip batches if resuming mid-epoch
            if epoch == start_epoch and batch_idx < (start_step * args.gradient_accumulation_steps) % len(dataloader):
                continue
            
            # Move to device
            batch = {k: v.cuda() for k, v in batch.items()}
            
            # Forward and backward
            try:
                losses = train_step(model, batch, loss_fn)
                loss = losses["loss"] / args.gradient_accumulation_steps
                loss.backward()
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"\n[OOM] CUDA out of memory at step {global_step}, clearing cache...")
                    torch.cuda.empty_cache()
                    continue
                raise
            
            accumulated_loss += losses["loss"].item()
            num_accumulated += 1
            
            # Gradient step
            if (batch_idx + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                
                optimizer.step()
                if lr_scheduler is not None:
                    lr_scheduler.step()
                optimizer.zero_grad()
                
                # Update schedulers
                sparsity_scheduler.step()
                temperature_scheduler.step()
                phase_scheduler.step()
                
                # Update mask temperatures
                temp = temperature_scheduler.get_temperature()
                for srde_layer in model.srde_layers.values():
                    srde_layer.mask_selector.set_temperature(temp)
                
                global_step += 1
                TRAINING_STATE["global_step"] = global_step
                
                # Logging
                if global_step % args.log_steps == 0:
                    avg_loss = accumulated_loss / num_accumulated
                    phase = phase_scheduler.phase_name
                    sparsity = sparsity_scheduler.get_sparsity()
                    
                    # Update best loss
                    is_best = avg_loss < TRAINING_STATE["best_loss"]
                    if is_best:
                        TRAINING_STATE["best_loss"] = avg_loss
                    
                    progress.set_postfix({
                        "loss": f"{avg_loss:.4f}",
                        "phase": phase,
                        "sparsity": f"{sparsity:.3f}",
                        "step": global_step
                    })
                    
                    if args.use_wandb and WANDB_AVAILABLE:
                        log_dict = {
                            "loss": avg_loss,
                            "phase": phase_scheduler.current_phase,
                            "sparsity": sparsity,
                            "temperature": temp,
                            "step": global_step
                        }
                        if lr_scheduler is not None:
                            log_dict["learning_rate"] = lr_scheduler.get_last_lr()[0]
                        wandb.log(log_dict)
                    
                    accumulated_loss = 0.0
                    num_accumulated = 0
                
                # Heartbeat
                if global_step % args.heartbeat_steps == 0:
                    write_heartbeat(
                        args.output_dir,
                        global_step,
                        losses["loss"].item(),
                        phase_scheduler.phase_name
                    )
                
                # Checkpointing
                if global_step % args.save_steps == 0:
                    avg_loss = losses["loss"].item()
                    is_best = avg_loss < TRAINING_STATE["best_loss"]
                    
                    save_checkpoint(
                        model, optimizer, lr_scheduler, schedulers,
                        global_step, epoch, avg_loss,
                        args.output_dir, is_best=is_best
                    )
                
                # Check max steps
                if args.max_steps > 0 and global_step >= args.max_steps:
                    break
        
        if args.max_steps > 0 and global_step >= args.max_steps:
            break
    
    #---------------------------------------------------------------------------
    # Final save
    #---------------------------------------------------------------------------
    print("\n" + "="*60)
    print("Training complete!")
    print("="*60)
    
    final_path = save_checkpoint(
        model, optimizer, lr_scheduler, schedulers,
        global_step, args.num_epochs, 0.0,
        args.output_dir
    )
    
    # Write completion marker
    with open(Path(args.output_dir) / "TRAINING_COMPLETE", "w") as f:
        f.write(json.dumps({
            "final_step": global_step,
            "final_checkpoint": final_path,
            "timestamp": datetime.now().isoformat(),
            "best_loss": TRAINING_STATE["best_loss"]
        }, indent=2))
    
    print(f"Final checkpoint: {final_path}")
    print(f"Best loss: {TRAINING_STATE['best_loss']:.4f}")
    
    if args.use_wandb and WANDB_AVAILABLE:
        wandb.finish()


if __name__ == "__main__":
    main()
