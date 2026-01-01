"""
SRDE v2.1: Advanced Training Features

New capabilities:
1. Adaptive Warmup - Faster sparsity schedule with dynamic adjustment
2. Expert-Specific Loss - Route-aware loss for better specialization
3. Curriculum Learning - Easy→Hard domain ordering
4. RL Fine-tuning - Reward-based training for correct reasoning
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import random
import math


# =============================================================================
# 1. ADAPTIVE WARMUP SCHEDULER
# =============================================================================

class AdaptiveSparsityScheduler:
    """
    Adaptive sparsity schedule that speeds up or slows down based on training dynamics.
    
    If loss is improving: accelerate sparsity reduction
    If loss is stalling: slow down sparsity reduction
    """
    
    def __init__(
        self,
        initial_sparsity: float = 0.05,
        target_sparsity: float = 0.01,
        base_warmup_steps: int = 10000,
        min_warmup_steps: int = 5000,
        max_warmup_steps: int = 50000,
        patience: int = 500,
        improvement_threshold: float = 0.001
    ):
        self.initial_sparsity = initial_sparsity
        self.target_sparsity = target_sparsity
        self.base_warmup_steps = base_warmup_steps
        self.current_warmup_steps = base_warmup_steps
        self.min_warmup = min_warmup_steps
        self.max_warmup = max_warmup_steps
        
        self.patience = patience
        self.improvement_threshold = improvement_threshold
        
        self.current_step = 0
        self.loss_history = []
        self.last_adjustment_step = 0
    
    def step(self, current_loss: Optional[float] = None):
        """Advance scheduler and optionally adjust based on loss."""
        self.current_step += 1
        
        if current_loss is not None:
            self.loss_history.append(current_loss)
            
            # Check for adjustment every `patience` steps
            if self.current_step - self.last_adjustment_step >= self.patience:
                self._maybe_adjust()
                self.last_adjustment_step = self.current_step
    
    def _maybe_adjust(self):
        """Adjust warmup schedule based on recent loss trend."""
        if len(self.loss_history) < self.patience:
            return
        
        recent = self.loss_history[-self.patience:]
        earlier = self.loss_history[-2*self.patience:-self.patience] if len(self.loss_history) >= 2*self.patience else recent
        
        recent_avg = sum(recent) / len(recent)
        earlier_avg = sum(earlier) / len(earlier)
        
        improvement = (earlier_avg - recent_avg) / earlier_avg
        
        if improvement > self.improvement_threshold:
            # Loss improving - accelerate sparsity reduction
            self.current_warmup_steps = max(
                self.min_warmup,
                int(self.current_warmup_steps * 0.9)
            )
            print(f"[AdaptiveSparsity] Loss improving, accelerating: warmup={self.current_warmup_steps}")
        elif improvement < -self.improvement_threshold:
            # Loss worsening - slow down
            self.current_warmup_steps = min(
                self.max_warmup,
                int(self.current_warmup_steps * 1.1)
            )
            print(f"[AdaptiveSparsity] Loss stalling, slowing: warmup={self.current_warmup_steps}")
    
    def get_sparsity(self) -> float:
        """Get current sparsity level."""
        if self.current_step >= self.current_warmup_steps:
            return self.target_sparsity
        
        progress = self.current_step / self.current_warmup_steps
        sparsity = self.initial_sparsity - (self.initial_sparsity - self.target_sparsity) * progress
        return max(self.target_sparsity, sparsity)


# =============================================================================
# 2. EXPERT-SPECIFIC LOSS
# =============================================================================

class ExpertSpecificLoss(nn.Module):
    """
    Route-aware loss function that:
    1. Weights loss by expert confidence
    2. Adds auxiliary losses for expert specialization
    3. Penalizes routing imbalance
    """
    
    def __init__(
        self,
        num_experts: int = 8,
        num_domains: int = 6,
        balance_weight: float = 0.1,
        specialization_weight: float = 0.05,
        diversity_weight: float = 0.01
    ):
        super().__init__()
        self.num_experts = num_experts
        self.num_domains = num_domains
        self.balance_weight = balance_weight
        self.specialization_weight = specialization_weight
        self.diversity_weight = diversity_weight
        
        # Track expert-domain affinity
        self.register_buffer(
            "expert_domain_counts",
            torch.zeros(num_experts, num_domains)
        )
    
    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        router_weights: torch.Tensor,
        selected_experts: torch.Tensor,
        domain_ids: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute expert-aware loss.
        
        Args:
            logits: Model predictions (B, S, V)
            labels: Ground truth (B, S)
            router_weights: Expert weights (B, S, top_k)
            selected_experts: Selected expert indices (B, S, top_k)
            domain_ids: Domain labels per example (B,)
        """
        # Standard cross-entropy
        ce_loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            labels.view(-1),
            ignore_index=-100
        )
        
        # Balance loss - penalize uneven expert usage
        expert_usage = torch.zeros(self.num_experts, device=logits.device)
        for i in range(self.num_experts):
            expert_usage[i] = (selected_experts == i).float().sum()
        expert_usage = expert_usage / expert_usage.sum().clamp(min=1)
        uniform = torch.ones_like(expert_usage) / self.num_experts
        balance_loss = F.kl_div(
            expert_usage.log().unsqueeze(0),
            uniform.unsqueeze(0),
            reduction='batchmean'
        )
        
        # Specialization loss - encourage high confidence routing
        confidence_loss = -router_weights.max(dim=-1).values.mean()
        
        # Diversity loss - encourage different experts for different tokens
        if router_weights.size(1) > 1:  # More than 1 token
            routing_entropy = -(router_weights * router_weights.log().clamp(min=-100)).sum(-1).mean()
            diversity_loss = -routing_entropy  # Negative because we want low entropy
        else:
            diversity_loss = torch.tensor(0.0, device=logits.device)
        
        # Update domain affinity tracking
        if domain_ids is not None:
            for b in range(selected_experts.size(0)):
                domain = domain_ids[b].item() if domain_ids[b].dim() == 0 else domain_ids[b][0].item()
                for expert_idx in selected_experts[b].unique():
                    if expert_idx >= 0 and expert_idx < self.num_experts:
                        if domain < self.num_domains:
                            self.expert_domain_counts[expert_idx, domain] += 1
        
        # Combine losses
        total_loss = (
            ce_loss +
            self.balance_weight * balance_loss +
            self.specialization_weight * confidence_loss +
            self.diversity_weight * diversity_loss
        )
        
        return {
            "loss": total_loss,
            "ce_loss": ce_loss,
            "balance_loss": balance_loss,
            "confidence_loss": confidence_loss,
            "diversity_loss": diversity_loss
        }


# =============================================================================
# 3. CURRICULUM LEARNING
# =============================================================================

@dataclass
class CurriculumConfig:
    """Configuration for curriculum learning."""
    strategy: str = "linear"  # linear, exponential, or competence
    warmup_epochs: int = 1
    difficulty_bins: int = 5
    min_difficulty: float = 0.0
    max_difficulty: float = 1.0


class CurriculumScheduler:
    """
    Easy→Hard curriculum scheduler.
    
    Strategies:
    - linear: Linearly increase difficulty over training
    - exponential: Slow start, fast ramp-up
    - competence: Increase based on model competence (loss)
    """
    
    def __init__(
        self,
        config: CurriculumConfig,
        total_steps: int,
        dataset_difficulties: Optional[torch.Tensor] = None
    ):
        self.config = config
        self.total_steps = total_steps
        self.warmup_steps = int(total_steps * config.warmup_epochs / 10)  # Assume 10 epochs
        self.current_step = 0
        
        # Pre-computed difficulties per example (0 = easy, 1 = hard)
        self.difficulties = dataset_difficulties
    
    def step(self):
        self.current_step += 1
    
    def get_difficulty_threshold(self) -> float:
        """Get current maximum difficulty threshold."""
        if self.current_step < self.warmup_steps:
            return self.config.min_difficulty  # Only easy examples during warmup
        
        progress = (self.current_step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
        progress = min(1.0, max(0.0, progress))
        
        if self.config.strategy == "linear":
            threshold = self.config.min_difficulty + progress * (self.config.max_difficulty - self.config.min_difficulty)
        elif self.config.strategy == "exponential":
            threshold = self.config.min_difficulty + (1 - math.exp(-3 * progress)) * (self.config.max_difficulty - self.config.min_difficulty)
        else:  # competence-based (implemented elsewhere)
            threshold = self.config.max_difficulty
        
        return threshold
    
    def filter_batch(self, batch_indices: torch.Tensor) -> torch.Tensor:
        """Filter batch to only include examples within difficulty threshold."""
        if self.difficulties is None:
            return batch_indices
        
        threshold = self.get_difficulty_threshold()
        mask = self.difficulties[batch_indices] <= threshold
        
        # Always keep at least some examples
        if mask.sum() < len(batch_indices) // 4:
            # If too few, keep the easiest quarter
            k = max(1, len(batch_indices) // 4)
            _, easiest = torch.topk(self.difficulties[batch_indices], k, largest=False)
            return batch_indices[easiest]
        
        return batch_indices[mask]
    
    def compute_difficulties(self, model, dataloader, tokenizer) -> torch.Tensor:
        """
        Pre-compute difficulty scores for all examples.
        Difficulty = perplexity under base model (higher = harder)
        """
        print("[Curriculum] Computing example difficulties...")
        difficulties = []
        
        model.eval()
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch['input_ids'].cuda()
                attention_mask = batch.get('attention_mask', torch.ones_like(input_ids)).cuda()
                
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
                loss = outputs.loss
                
                # Per-example loss as difficulty
                difficulties.append(loss.item())
        
        model.train()
        
        # Normalize to [0, 1]
        difficulties = torch.tensor(difficulties)
        difficulties = (difficulties - difficulties.min()) / (difficulties.max() - difficulties.min() + 1e-8)
        
        print(f"[Curriculum] Computed difficulties for {len(difficulties)} examples")
        return difficulties


# =============================================================================
# 4. RL FINE-TUNING (REWARD-BASED)
# =============================================================================

class RewardModel(nn.Module):
    """
    Simple reward model for RL fine-tuning.
    Rewards correct reasoning patterns.
    """
    
    def __init__(self, hidden_size: int):
        super().__init__()
        self.reward_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 4),
            nn.GELU(),
            nn.Linear(hidden_size // 4, 1)
        )
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Compute reward for each position.
        
        Args:
            hidden_states: (B, S, D)
        Returns:
            rewards: (B, S)
        """
        return self.reward_head(hidden_states).squeeze(-1)


class RLFineTuner:
    """
    RL fine-tuning using PPO-style updates.
    
    Reward signals:
    1. Correctness reward (from verifier)
    2. Reasoning quality (from reward model)
    3. Expert utilization reward
    """
    
    def __init__(
        self,
        model,
        reward_model: RewardModel,
        learning_rate: float = 1e-6,
        ppo_epochs: int = 4,
        clip_epsilon: float = 0.2,
        gamma: float = 0.99,
        gae_lambda: float = 0.95
    ):
        self.model = model
        self.reward_model = reward_model
        self.ppo_epochs = ppo_epochs
        self.clip_epsilon = clip_epsilon
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    
    def compute_correctness_reward(
        self,
        generated_answer: str,
        correct_answer: str
    ) -> float:
        """Binary reward for correct answer."""
        # Simple exact match for now
        generated_clean = generated_answer.strip().lower()
        correct_clean = correct_answer.strip().lower()
        
        if generated_clean == correct_clean:
            return 1.0
        
        # Partial credit if answer is contained
        if correct_clean in generated_clean:
            return 0.5
        
        return 0.0
    
    def compute_expert_utilization_reward(
        self,
        router_weights: torch.Tensor,
        selected_experts: torch.Tensor
    ) -> float:
        """Reward for good expert utilization."""
        # Reward high confidence routing
        confidence = router_weights.max(dim=-1).values.mean().item()
        
        # Penalize if always using same expert
        unique_experts = selected_experts.unique().numel()
        diversity = unique_experts / self.model.config.num_experts if hasattr(self.model, 'config') else unique_experts / 8
        
        return 0.5 * confidence + 0.5 * diversity
    
    def ppo_step(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        old_log_probs: torch.Tensor,
        rewards: torch.Tensor,
        advantages: torch.Tensor
    ) -> Dict[str, float]:
        """
        Perform PPO update step.
        """
        total_loss = 0.0
        
        for _ in range(self.ppo_epochs):
            # Forward pass
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            
            # Compute new log probs
            new_log_probs = F.log_softmax(logits, dim=-1)
            action_log_probs = new_log_probs.gather(-1, input_ids.unsqueeze(-1)).squeeze(-1)
            
            # PPO clipped objective
            ratio = torch.exp(action_log_probs - old_log_probs)
            clipped_ratio = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon)
            
            policy_loss = -torch.min(ratio * advantages, clipped_ratio * advantages).mean()
            
            # Value loss (if using value head)
            # value_loss = F.mse_loss(values, rewards)
            
            loss = policy_loss
            
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
        
        return {
            "ppo_loss": total_loss / self.ppo_epochs
        }


# =============================================================================
# 5. DATA PIPELINE CONFIG (for 50B tokens)
# =============================================================================

@dataclass
class LargeScaleDataConfig:
    """Configuration for 50B+ token training."""
    
    # Target sizes
    total_tokens: int = 50_000_000_000  # 50B
    
    # Domain distribution (balanced)
    domains: Dict[str, float] = None
    
    # Data sources
    synthetic_ratio: float = 0.7  # 70% synthetic, 30% curated
    
    # Generation settings
    generator_model: str = "Qwen/Qwen2.5-72B-Instruct"
    generation_temperature: float = 0.7
    
    # Verification
    verify_answers: bool = True
    verification_model: str = "same"  # Use same model or separate
    
    # Storage
    output_format: str = "jsonl"
    shard_size_gb: float = 5.0
    
    def __post_init__(self):
        if self.domains is None:
            self.domains = {
                "math": 0.167,
                "code": 0.167,
                "science": 0.167,
                "logic": 0.167,
                "planning": 0.167,
                "abstract": 0.165
            }
    
    def get_tokens_per_domain(self) -> Dict[str, int]:
        return {
            domain: int(self.total_tokens * ratio)
            for domain, ratio in self.domains.items()
        }


print("SRDE v2.1 Advanced Features loaded!")
