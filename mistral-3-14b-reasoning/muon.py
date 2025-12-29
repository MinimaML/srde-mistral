"""
Muon Optimizer for SRDE

Muon (Momentum Orthogonalized Update) optimizer by Keller Jordan.
~35% faster training than AdamW for transformer hidden weights.

Key insight: Orthogonalizes SGD-momentum updates via Newton-Schulz iteration
to better utilize parameter space.

Usage:
- Use Muon for 2D hidden weights (attention, FFN projections, SRDE deltas)
- Use AdamW for embeddings, layer norms, biases, 1D params
"""

import torch
from torch.optim import Optimizer
from typing import Iterable, Tuple, Optional
import math


def zeropower_via_newtonschulz5(G: torch.Tensor, steps: int = 5) -> torch.Tensor:
    """
    Newton-Schulz iteration to compute the orthogonal component of G.
    
    Finds the nearest semi-orthogonal matrix to G.
    This is the core of Muon's efficiency gain.
    """
    assert len(G.shape) == 2
    
    a, b, c = (3.4445, -4.7750, 2.0315)
    
    X = G.bfloat16() if G.dtype != torch.bfloat16 else G
    
    # Normalize
    X = X / (X.norm() + 1e-7)
    
    # Newton-Schulz iterations
    for _ in range(steps):
        A = X @ X.T
        B = A @ X
        X = a * X + b * B + c * A @ B
    
    return X.to(G.dtype)


class Muon(Optimizer):
    """
    Muon - Momentum Orthogonalized Update optimizer.
    
    Designed for 2D weight matrices in neural networks.
    Uses Newton-Schulz iteration to orthogonalize momentum updates.
    
    Args:
        params: Iterable of parameters to optimize
        lr: Learning rate (default: 0.02)
        momentum: Momentum factor (default: 0.95)
        nesterov: Use Nesterov momentum (default: True)
        ns_steps: Newton-Schulz iteration steps (default: 5)
        weight_decay: Weight decay (default: 0.0)
    
    Note:
        - Only use for 2D parameters (weight matrices)
        - For 1D params, biases, embeddings, use AdamW instead
    """
    
    def __init__(
        self,
        params: Iterable,
        lr: float = 0.02,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 5,
        weight_decay: float = 0.0
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0 or momentum > 1.0:
            raise ValueError(f"Invalid momentum: {momentum}")
        if ns_steps < 1:
            raise ValueError(f"Invalid ns_steps: {ns_steps}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay: {weight_decay}")
        
        defaults = dict(
            lr=lr,
            momentum=momentum,
            nesterov=nesterov,
            ns_steps=ns_steps,
            weight_decay=weight_decay
        )
        super().__init__(params, defaults)
    
    @torch.no_grad()
    def step(self, closure=None):
        """Perform a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']
            nesterov = group['nesterov']
            ns_steps = group['ns_steps']
            weight_decay = group['weight_decay']
            
            for p in group['params']:
                if p.grad is None:
                    continue
                
                # Get gradient
                g = p.grad
                
                # Weight decay
                if weight_decay != 0:
                    g = g.add(p, alpha=weight_decay)
                
                # Only apply Newton-Schulz to 2D params
                if g.dim() == 2:
                    state = self.state[p]
                    
                    # Initialize momentum buffer
                    if 'momentum_buffer' not in state:
                        state['momentum_buffer'] = torch.zeros_like(g)
                    
                    buf = state['momentum_buffer']
                    
                    # Momentum update
                    buf.mul_(momentum).add_(g)
                    
                    if nesterov:
                        g = g.add(buf, alpha=momentum)
                    else:
                        g = buf
                    
                    # Newton-Schulz orthogonalization
                    g = zeropower_via_newtonschulz5(g, steps=ns_steps)
                    
                    # Scale update (muP-style scaling)
                    scale = max(1, g.size(0) / g.size(1)) ** 0.5
                    
                    # Apply update
                    p.add_(g, alpha=-lr * scale)
                
                else:
                    # For 1D params, just use SGD with momentum
                    state = self.state[p]
                    
                    if 'momentum_buffer' not in state:
                        state['momentum_buffer'] = torch.zeros_like(g)
                    
                    buf = state['momentum_buffer']
                    buf.mul_(momentum).add_(g)
                    
                    if nesterov:
                        g = g.add(buf, alpha=momentum)
                    else:
                        g = buf
                    
                    p.add_(g, alpha=-lr)
        
        return loss


def get_muon_param_groups(
    model,
    muon_lr: float = 0.02,
    adamw_lr: float = 1e-4,
    muon_momentum: float = 0.95,
    adamw_betas: Tuple[float, float] = (0.9, 0.95),
    weight_decay: float = 0.01
) -> Tuple[list, list]:
    """
    Split model parameters into Muon and AdamW groups.
    
    Muon: 2D hidden weights (projections, FFN, SRDE deltas)
    AdamW: Embeddings, layer norms, biases, 1D params
    
    Args:
        model: The model to get parameters from
        muon_lr: Learning rate for Muon
        adamw_lr: Learning rate for AdamW
        
    Returns:
        (muon_params, adamw_params): Two lists of parameter dicts
    """
    muon_params = []
    adamw_params = []
    
    # Keywords that indicate AdamW should be used
    adamw_keywords = [
        'embed', 'embedding',
        'norm', 'ln', 'layernorm', 'layer_norm', 'rmsnorm',
        'bias',
        'lm_head', 'head', 'classifier',
        'wte', 'wpe',  # GPT-style embeddings
    ]
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        
        name_lower = name.lower()
        
        # Check if this should use AdamW
        use_adamw = False
        
        # 1D params always use AdamW
        if param.dim() == 1:
            use_adamw = True
        
        # Check for AdamW keywords
        for kw in adamw_keywords:
            if kw in name_lower:
                use_adamw = True
                break
        
        if use_adamw:
            adamw_params.append({
                'params': [param],
                'lr': adamw_lr,
                'betas': adamw_betas,
                'weight_decay': weight_decay if 'bias' not in name_lower else 0.0,
                'name': name
            })
        else:
            muon_params.append({
                'params': [param],
                'lr': muon_lr,
                'momentum': muon_momentum,
                'weight_decay': weight_decay,
                'name': name
            })
    
    return muon_params, adamw_params


class MuonAdamW:
    """
    Combined optimizer that uses Muon for 2D weights and AdamW for the rest.
    
    This is the recommended way to use Muon for LLM training.
    """
    
    def __init__(
        self,
        model,
        muon_lr: float = 0.02,
        adamw_lr: float = 1e-4,
        muon_momentum: float = 0.95,
        adamw_betas: Tuple[float, float] = (0.9, 0.95),
        weight_decay: float = 0.01,
        ns_steps: int = 5
    ):
        muon_groups, adamw_groups = get_muon_param_groups(
            model, muon_lr, adamw_lr, muon_momentum, adamw_betas, weight_decay
        )
        
        # Count parameters
        muon_count = sum(p['params'][0].numel() for p in muon_groups)
        adamw_count = sum(p['params'][0].numel() for p in adamw_groups)
        
        print(f"[MuonAdamW] Muon params: {muon_count:,} | AdamW params: {adamw_count:,}")
        
        # Create optimizers
        if muon_groups:
            self.muon = Muon(
                [p['params'][0] for p in muon_groups],
                lr=muon_lr,
                momentum=muon_momentum,
                weight_decay=weight_decay,
                ns_steps=ns_steps
            )
        else:
            self.muon = None
        
        if adamw_groups:
            from torch.optim import AdamW
            self.adamw = AdamW(
                adamw_groups,
                lr=adamw_lr,
                betas=adamw_betas,
                weight_decay=weight_decay,
                fused=torch.cuda.is_available()  # Use fused AdamW if available
            )
        else:
            self.adamw = None
    
    def zero_grad(self, set_to_none: bool = True):
        """Zero gradients for both optimizers."""
        if self.muon:
            self.muon.zero_grad(set_to_none=set_to_none)
        if self.adamw:
            self.adamw.zero_grad(set_to_none=set_to_none)
    
    def step(self, closure=None):
        """Step both optimizers."""
        loss = None
        if self.muon:
            loss = self.muon.step(closure)
        if self.adamw:
            loss = self.adamw.step(closure if loss is None else None)
        return loss
    
    def state_dict(self) -> dict:
        """Get combined state dict."""
        return {
            'muon': self.muon.state_dict() if self.muon else None,
            'adamw': self.adamw.state_dict() if self.adamw else None
        }
    
    def load_state_dict(self, state_dict: dict):
        """Load combined state dict."""
        if self.muon and state_dict.get('muon'):
            self.muon.load_state_dict(state_dict['muon'])
        if self.adamw and state_dict.get('adamw'):
            self.adamw.load_state_dict(state_dict['adamw'])
