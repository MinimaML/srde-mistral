"""
SRDE Loss Functions

HARDENED VERSION with:
- Numerical stability
- Edge case handling
- NaN protection
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict
import logging

from config import SRDEConfig

logger = logging.getLogger(__name__)


def safe_mean(tensor: torch.Tensor, dim=None) -> torch.Tensor:
    """Mean with NaN protection."""
    if tensor.numel() == 0:
        return torch.tensor(0.0, device=tensor.device, dtype=tensor.dtype)
    result = tensor.mean(dim=dim) if dim is not None else tensor.mean()
    if torch.isnan(result).any():
        logger.warning("NaN in mean, returning 0")
        return torch.zeros_like(result)
    return result


def load_balance_loss(
    router_logits: torch.Tensor,
    num_experts: int,
    eps: float = 1e-8
) -> torch.Tensor:
    """
    Encourage uniform expert utilization.
    
    HARDENED: Handles empty tensors, NaN protection.
    """
    if router_logits is None or router_logits.numel() == 0:
        return torch.tensor(0.0)
    
    if num_experts <= 0:
        logger.warning(f"Invalid num_experts: {num_experts}")
        return torch.tensor(0.0, device=router_logits.device)
    
    # Clamp for numerical stability
    router_logits = torch.clamp(router_logits, min=-50, max=50)
    
    # Convert to probabilities
    router_probs = F.softmax(router_logits, dim=-1)
    
    # Average probability per expert
    # Handle variable dimensions
    if router_probs.dim() == 3:
        avg_probs = router_probs.mean(dim=[0, 1])
    elif router_probs.dim() == 2:
        avg_probs = router_probs.mean(dim=0)
    else:
        avg_probs = router_probs
    
    # Target is uniform
    target = 1.0 / num_experts
    
    # MSE loss
    loss = F.mse_loss(avg_probs, torch.full_like(avg_probs, target))
    
    # NaN check
    if torch.isnan(loss):
        logger.warning("NaN in load_balance_loss")
        return torch.tensor(0.0, device=router_logits.device)
    
    return loss


def orthogonality_loss(
    masks: torch.Tensor,
    eps: float = 1e-8
) -> torch.Tensor:
    """
    Encourage diverse expert masks.
    
    HARDENED: Handles edge cases, numerical stability.
    """
    if masks is None or masks.numel() == 0:
        return torch.tensor(0.0)
    
    num_experts = masks.size(0)
    num_params = masks.size(1)
    
    if num_experts < 2:
        return torch.tensor(0.0, device=masks.device)
    
    if num_params == 0:
        return torch.tensor(0.0, device=masks.device)
    
    # Convert to float for computation
    mask_float = masks.float()
    
    # Gram matrix
    gram = mask_float @ mask_float.T
    
    # Off-diagonal mask
    off_diag = 1.0 - torch.eye(num_experts, device=masks.device)
    
    # Overlap sum
    overlap_sum = (gram * off_diag).sum()
    
    # Normalize
    num_pairs = num_experts * (num_experts - 1)
    loss = overlap_sum / (num_pairs * num_params + eps)
    
    # NaN check
    if torch.isnan(loss):
        logger.warning("NaN in orthogonality_loss")
        return torch.tensor(0.0, device=masks.device)
    
    return loss


def sparsity_loss(
    experts: nn.ModuleList,
    target_l1: float = 0.01,
    eps: float = 1e-8
) -> torch.Tensor:
    """
    Regularize delta magnitudes.
    
    HARDENED: Handles empty experts, error recovery.
    """
    if experts is None or len(experts) == 0:
        return torch.tensor(0.0)
    
    total_l1 = None
    count = 0
    
    for expert in experts:
        try:
            delta = expert.get_weighted_delta()
            
            if delta is None or delta.numel() == 0:
                continue
            
            # Initialize on first valid delta
            if total_l1 is None:
                total_l1 = delta.abs().mean()
            else:
                total_l1 = total_l1 + delta.abs().mean()
            
            count += 1
            
        except Exception as e:
            logger.warning(f"Error getting delta from expert: {e}")
            continue
    
    if count == 0 or total_l1 is None:
        return torch.tensor(0.0)
    
    avg_l1 = total_l1 / count
    
    # Only penalize if above target
    loss = F.relu(avg_l1 - target_l1)
    
    # NaN check
    if torch.isnan(loss):
        logger.warning("NaN in sparsity_loss")
        return torch.tensor(0.0, device=total_l1.device)
    
    return loss


def diversity_loss(
    router_logits: torch.Tensor,
    temperature: float = 1.0,
    eps: float = 1e-8
) -> torch.Tensor:
    """
    Encourage moderate routing entropy.
    
    HARDENED: Numerical stability, edge cases.
    """
    if router_logits is None or router_logits.numel() == 0:
        return torch.tensor(0.0)
    
    if temperature <= 0:
        temperature = 1.0
    
    # Clamp for stability
    router_logits = torch.clamp(router_logits, min=-50, max=50)
    
    # Compute entropy
    probs = F.softmax(router_logits / temperature, dim=-1)
    log_probs = F.log_softmax(router_logits / temperature, dim=-1)
    
    # Entropy: -sum(p * log(p))
    entropy = -(probs * log_probs).sum(dim=-1)
    avg_entropy = safe_mean(entropy)
    
    # Target entropy
    num_experts = router_logits.size(-1)
    uniform_entropy = torch.log(torch.tensor(float(max(num_experts, 1)), device=router_logits.device))
    target_entropy = uniform_entropy * 0.5
    
    # Penalize deviation
    loss = (avg_entropy - target_entropy).abs()
    
    # NaN check
    if torch.isnan(loss):
        logger.warning("NaN in diversity_loss")
        return torch.tensor(0.0, device=router_logits.device)
    
    return loss


class SRDELoss(nn.Module):
    """
    Combined SRDE loss.
    
    HARDENED: Graceful degradation, detailed logging.
    """
    
    def __init__(self, config: SRDEConfig):
        super().__init__()
        self.config = config
        self._call_count = 0
    
    def forward(
        self,
        task_loss: torch.Tensor,
        router_logits: Optional[torch.Tensor] = None,
        masks: Optional[torch.Tensor] = None,
        experts: Optional[nn.ModuleList] = None
    ) -> Dict[str, torch.Tensor]:
        """Compute total loss."""
        self._call_count += 1
        
        device = task_loss.device if task_loss is not None else torch.device('cpu')
        
        # Initialize losses
        losses = {
            'task_loss': task_loss if task_loss is not None else torch.tensor(0.0, device=device),
            'load_balance': torch.tensor(0.0, device=device),
            'orthogonality': torch.tensor(0.0, device=device),
            'sparsity': torch.tensor(0.0, device=device),
            'diversity': torch.tensor(0.0, device=device),
        }
        
        # Compute individual losses with error handling
        try:
            if router_logits is not None:
                losses['load_balance'] = load_balance_loss(
                    router_logits, self.config.num_experts
                ).to(device)
        except Exception as e:
            logger.warning(f"load_balance_loss failed: {e}")
        
        try:
            if masks is not None:
                losses['orthogonality'] = orthogonality_loss(masks).to(device)
        except Exception as e:
            logger.warning(f"orthogonality_loss failed: {e}")
        
        try:
            if experts is not None:
                losses['sparsity'] = sparsity_loss(experts).to(device)
        except Exception as e:
            logger.warning(f"sparsity_loss failed: {e}")
        
        try:
            if router_logits is not None:
                losses['diversity'] = diversity_loss(router_logits).to(device)
        except Exception as e:
            logger.warning(f"diversity_loss failed: {e}")
        
        # Weighted sum
        total = (
            losses['task_loss'] +
            self.config.lambda_load_balance * losses['load_balance'] +
            self.config.lambda_orthogonality * losses['orthogonality'] +
            self.config.lambda_sparsity * losses['sparsity']
        )
        
        # Final NaN check
        if torch.isnan(total):
            logger.error("NaN in total loss, falling back to task_loss only")
            total = losses['task_loss']
        
        losses['total_loss'] = total
        
        return losses
