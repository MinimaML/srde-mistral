"""
SRDE Training Schedulers

HARDENED VERSION with:
- Input validation
- Bounds checking
- State validation
"""
import torch
from typing import Optional
import logging

from config import SRDEConfig

logger = logging.getLogger(__name__)


class SparsityScheduler:
    """
    Progressive sparsity annealing.
    
    HARDENED: Bounds validation, state integrity.
    """
    
    def __init__(
        self,
        initial_sparsity: float = 0.05,
        target_sparsity: float = 0.01,
        warmup_steps: int = 1000,
        anneal_type: str = "linear"
    ):
        # Validate inputs
        if not (0 < initial_sparsity <= 1):
            raise ValueError(f"initial_sparsity must be in (0, 1], got {initial_sparsity}")
        if not (0 < target_sparsity <= 1):
            raise ValueError(f"target_sparsity must be in (0, 1], got {target_sparsity}")
        if warmup_steps < 0:
            raise ValueError(f"warmup_steps must be non-negative, got {warmup_steps}")
        if anneal_type not in ("linear", "cosine"):
            raise ValueError(f"anneal_type must be 'linear' or 'cosine', got {anneal_type}")
        
        self.initial_sparsity = initial_sparsity
        self.target_sparsity = target_sparsity
        self.warmup_steps = max(1, warmup_steps)  # Avoid division by zero
        self.anneal_type = anneal_type
        
        self.current_step = 0
    
    def step(self):
        """Advance by one step."""
        self.current_step += 1
    
    def get_sparsity(self, step: Optional[int] = None) -> float:
        """Get sparsity for current or specified step."""
        if step is None:
            step = self.current_step
        
        step = max(0, step)  # Clamp to non-negative
        
        if step >= self.warmup_steps:
            return self.target_sparsity
        
        progress = step / self.warmup_steps
        
        if self.anneal_type == "cosine":
            import math
            progress = 0.5 * (1 + math.cos(math.pi * (1 - progress)))
        
        sparsity = self.initial_sparsity - (
            self.initial_sparsity - self.target_sparsity
        ) * progress
        
        # Clamp to valid range
        return max(self.target_sparsity, min(self.initial_sparsity, sparsity))
    
    def prune_masks(
        self,
        masks: torch.Tensor,
        importance_scores: torch.Tensor,
        target_sparsity: Optional[float] = None
    ) -> torch.Tensor:
        """Prune least important positions."""
        if masks is None or masks.numel() == 0:
            return masks
        
        if importance_scores is None or importance_scores.numel() == 0:
            return masks
        
        if target_sparsity is None:
            target_sparsity = self.get_sparsity()
        
        # Validate
        target_sparsity = max(0.0001, min(1.0, target_sparsity))
        
        num_params = masks.size(1)
        target_count = max(1, int(num_params * target_sparsity))
        
        pruned_masks = masks.clone()
        
        for i in range(masks.size(0)):
            current_count = masks[i].sum().item()
            
            if current_count > target_count:
                try:
                    mask_indices = masks[i].nonzero(as_tuple=True)[0]
                    
                    if len(mask_indices) == 0:
                        continue
                    
                    num_to_keep = target_count
                    num_to_prune = len(mask_indices) - num_to_keep
                    
                    if num_to_prune > 0 and i < importance_scores.size(0):
                        scores = importance_scores[i]
                        
                        # Get indices of lowest importance
                        _, prune_order = torch.topk(
                            scores[:len(mask_indices)],
                            min(num_to_prune, len(scores)),
                            largest=False
                        )
                        
                        # Clear pruned positions
                        actual_indices = mask_indices[prune_order]
                        pruned_masks[i, actual_indices] = False
                        
                except Exception as e:
                    logger.warning(f"Pruning failed for expert {i}: {e}")
                    continue
        
        return pruned_masks
    
    def state_dict(self) -> dict:
        return {
            'current_step': self.current_step,
            'initial_sparsity': self.initial_sparsity,
            'target_sparsity': self.target_sparsity,
            'warmup_steps': self.warmup_steps,
            'anneal_type': self.anneal_type
        }
    
    def load_state_dict(self, state: dict):
        self.current_step = state.get('current_step', 0)
        self.initial_sparsity = state.get('initial_sparsity', self.initial_sparsity)
        self.target_sparsity = state.get('target_sparsity', self.target_sparsity)
        self.warmup_steps = max(1, state.get('warmup_steps', self.warmup_steps))
        self.anneal_type = state.get('anneal_type', self.anneal_type)


class TemperatureScheduler:
    """
    Temperature annealing for Gumbel-softmax.
    
    HARDENED: Bounds validation.
    """
    
    def __init__(
        self,
        initial_temp: float = 1.0,
        min_temp: float = 0.1,
        anneal_steps: int = 2000,
        anneal_type: str = "exponential"
    ):
        if initial_temp <= 0:
            raise ValueError(f"initial_temp must be positive, got {initial_temp}")
        if min_temp <= 0:
            raise ValueError(f"min_temp must be positive, got {min_temp}")
        if anneal_steps <= 0:
            anneal_steps = 1
        if anneal_type not in ("exponential", "linear"):
            anneal_type = "exponential"
        
        self.initial_temp = initial_temp
        self.min_temp = min_temp
        self.anneal_steps = anneal_steps
        self.anneal_type = anneal_type
        
        self.current_step = 0
    
    def step(self):
        self.current_step += 1
    
    def get_temperature(self, step: Optional[int] = None) -> float:
        if step is None:
            step = self.current_step
        
        step = max(0, step)
        
        if step >= self.anneal_steps:
            return self.min_temp
        
        progress = step / self.anneal_steps
        
        if self.anneal_type == "exponential":
            import math
            try:
                decay_rate = math.log(self.min_temp / self.initial_temp)
                temp = self.initial_temp * math.exp(decay_rate * progress)
            except (ValueError, OverflowError):
                # Fallback to linear if exponential fails
                temp = self.initial_temp - (self.initial_temp - self.min_temp) * progress
        else:
            temp = self.initial_temp - (self.initial_temp - self.min_temp) * progress
        
        return max(self.min_temp, min(self.initial_temp, temp))
    
    def state_dict(self) -> dict:
        return {
            'current_step': self.current_step,
            'initial_temp': self.initial_temp,
            'min_temp': self.min_temp,
            'anneal_steps': self.anneal_steps
        }
    
    def load_state_dict(self, state: dict):
        self.current_step = state.get('current_step', 0)
        self.initial_temp = state.get('initial_temp', self.initial_temp)
        self.min_temp = state.get('min_temp', self.min_temp)
        self.anneal_steps = max(1, state.get('anneal_steps', self.anneal_steps))


class PhaseScheduler:
    """
    4-phase training procedure manager.
    
    HARDENED: State validation.
    """
    
    def __init__(self, config: SRDEConfig):
        if config is None:
            raise ValueError("config cannot be None")
        
        self.config = config
        self.current_step = 0
        
        # Phase boundaries
        self.phase1_end = max(0, config.phase1_warmup_steps)
        self.phase2_end = self.phase1_end + max(0, config.phase2_mask_steps)
        self.phase3_end = self.phase2_end + max(0, config.phase3_delta_steps)
    
    @property
    def current_phase(self) -> int:
        """Get current phase (1-4)."""
        if self.current_step < self.phase1_end:
            return 1
        elif self.current_step < self.phase2_end:
            return 2
        elif self.current_step < self.phase3_end:
            return 3
        else:
            return 4
    
    @property
    def phase_name(self) -> str:
        names = {
            1: "Warmup",
            2: "Mask Learning",
            3: "Delta Training",
            4: "Joint Fine-tuning"
        }
        return names.get(self.current_phase, "Unknown")
    
    @property
    def phase_progress(self) -> float:
        """Progress within current phase (0.0 to 1.0)."""
        phase = self.current_phase
        
        if phase == 1:
            start, end = 0, self.phase1_end
        elif phase == 2:
            start, end = self.phase1_end, self.phase2_end
        elif phase == 3:
            start, end = self.phase2_end, self.phase3_end
        else:
            return 1.0  # Phase 4 has no end
        
        duration = max(1, end - start)
        return min(1.0, max(0.0, (self.current_step - start) / duration))
    
    def step(self):
        self.current_step += 1
    
    def get_trainable_groups(self, srde_layers: dict) -> dict:
        """Get parameters to train in current phase."""
        phase = self.current_phase
        groups = {}
        
        if srde_layers is None:
            return groups
        
        for layer_idx, layer in srde_layers.items():
            try:
                if phase == 1:
                    pass  # Nothing trainable in warmup
                elif phase == 2:
                    groups[f'layer_{layer_idx}_mask'] = list(
                        layer.mask_selector.parameters()
                    )
                elif phase == 3:
                    groups[f'layer_{layer_idx}_vocabulary'] = list(
                        layer.vocabulary.parameters()
                    )
                    for i, expert in enumerate(layer.experts):
                        groups[f'layer_{layer_idx}_expert_{i}'] = list(
                            expert.parameters()
                        )
                else:  # Phase 4
                    groups[f'layer_{layer_idx}_all'] = list(layer.parameters())
            except Exception as e:
                logger.warning(f"Error getting params for layer {layer_idx}: {e}")
        
        return groups
    
    def should_update_masks(self) -> bool:
        """Whether to update masks this step."""
        return self.current_phase >= 3 and self.current_step % 100 == 0
    
    def state_dict(self) -> dict:
        return {'current_step': self.current_step}
    
    def load_state_dict(self, state: dict):
        self.current_step = max(0, state.get('current_step', 0))
