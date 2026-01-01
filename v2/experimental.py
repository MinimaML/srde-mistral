"""
SRDE v2.2: Experimental Innovations

Novel ideas that push SRDE beyond standard MoE:

1. Temporal Experts - Route based on position in context (recent vs old)
2. Uncertainty Routing - Route uncertain tokens to more experts
3. Expert Composition - Dynamically combine experts into synthetic experts
4. Memory Experts - Experts with persistent key-value memory
5. Recursive Experts - Experts that call themselves for hard problems
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import math


# =============================================================================
# 1. TEMPORAL EXPERTS - Position-aware routing
# =============================================================================

class TemporalRouter(nn.Module):
    """
    Routes tokens based on their position in context.
    
    Intuition: Recent tokens need different processing than old context.
    - Positions 0-512: "Memory" experts (recall)
    - Positions 512-1536: "Reasoning" experts (compute)
    - Positions 1536+: "Output" experts (generate)
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_experts: int,
        num_temporal_zones: int = 3,
        max_position: int = 4096
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.num_zones = num_temporal_zones
        self.max_position = max_position
        
        # Zone boundaries (learnable)
        self.zone_boundaries = nn.Parameter(
            torch.linspace(0, max_position, num_temporal_zones + 1)
        )
        
        # Per-zone routing weights
        self.zone_routers = nn.ModuleList([
            nn.Linear(hidden_size, num_experts)
            for _ in range(num_temporal_zones)
        ])
        
        # Zone blending (soft transitions)
        self.zone_temp = nn.Parameter(torch.tensor(100.0))
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            hidden_states: (B, S, D)
            positions: (B, S) or (S,) position indices
        """
        B, S, D = hidden_states.shape
        
        if positions.dim() == 1:
            positions = positions.unsqueeze(0).expand(B, -1)
        
        # Compute zone membership (soft)
        zone_weights = []
        for i in range(self.num_zones):
            left = self.zone_boundaries[i]
            right = self.zone_boundaries[i + 1]
            
            # Soft membership with sigmoid
            in_zone = torch.sigmoid(
                self.zone_temp * (positions - left)
            ) * torch.sigmoid(
                self.zone_temp * (right - positions)
            )
            zone_weights.append(in_zone)
        
        zone_weights = torch.stack(zone_weights, dim=-1)  # (B, S, num_zones)
        zone_weights = zone_weights / zone_weights.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        
        # Compute routing per zone
        router_logits = torch.zeros(B, S, self.num_experts, device=hidden_states.device)
        
        for i, router in enumerate(self.zone_routers):
            zone_logits = router(hidden_states)  # (B, S, E)
            router_logits += zone_weights[:, :, i:i+1] * zone_logits
        
        router_weights = F.softmax(router_logits, dim=-1)
        selected = torch.topk(router_weights, k=2, dim=-1)
        
        return selected.values, selected.indices


# =============================================================================
# 2. UNCERTAINTY ROUTING - More experts for uncertain tokens
# =============================================================================

class UncertaintyRouter(nn.Module):
    """
    Routes based on model uncertainty.
    
    High uncertainty → Use more experts (hedging bets)
    Low uncertainty → Use fewer experts (confident)
    
    Uncertainty estimated from hidden state variance.
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_experts: int,
        min_k: int = 1,
        max_k: int = 4,
        uncertainty_threshold: float = 0.5
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.min_k = min_k
        self.max_k = max_k
        self.uncertainty_threshold = uncertainty_threshold
        
        self.router = nn.Linear(hidden_size, num_experts)
        
        # Uncertainty estimator
        self.uncertainty_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 4),
            nn.GELU(),
            nn.Linear(hidden_size // 4, 1),
            nn.Sigmoid()
        )
    
    def forward(
        self,
        hidden_states: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            weights: Expert weights
            indices: Selected expert indices
            k_values: Number of experts per token
        """
        B, S, D = hidden_states.shape
        
        # Estimate uncertainty
        uncertainty = self.uncertainty_head(hidden_states).squeeze(-1)  # (B, S)
        
        # Compute k per token based on uncertainty
        k_float = self.min_k + (self.max_k - self.min_k) * uncertainty
        k_values = k_float.round().long().clamp(self.min_k, self.max_k)
        
        # Router logits
        logits = self.router(hidden_states)  # (B, S, E)
        
        # Variable top-k selection
        weights_list = []
        indices_list = []
        
        for b in range(B):
            for s in range(S):
                k = k_values[b, s].item()
                topk = torch.topk(logits[b, s], k=k)
                
                # Pad to max_k for batching
                w = F.pad(topk.values, (0, self.max_k - k), value=0)
                i = F.pad(topk.indices, (0, self.max_k - k), value=-1)
                
                weights_list.append(w)
                indices_list.append(i)
        
        weights = torch.stack(weights_list).view(B, S, self.max_k)
        indices = torch.stack(indices_list).view(B, S, self.max_k)
        
        # Normalize weights
        weights = F.softmax(weights, dim=-1)
        
        return weights, indices, k_values


# =============================================================================
# 3. EXPERT COMPOSITION - Combine experts dynamically
# =============================================================================

class ComposableExperts(nn.Module):
    """
    Dynamically compose experts into new synthetic experts.
    
    Instead of: output = sum(w_i * expert_i(x))
    We do: output = composed_expert(x) where composed = f(expert_1, expert_2, ...)
    
    This allows emergent capabilities from combinations.
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_base_experts: int,
        composition_dim: int = 64
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_base = num_base_experts
        
        # Base expert embeddings
        self.expert_embeddings = nn.Parameter(
            torch.randn(num_base_experts, composition_dim)
        )
        
        # Composition network
        self.composer = nn.Sequential(
            nn.Linear(composition_dim * 2, composition_dim),
            nn.GELU(),
            nn.Linear(composition_dim, composition_dim)
        )
        
        # Decode composed embedding to expert weights
        self.decoder = nn.Linear(composition_dim, num_base_experts)
    
    def compose(
        self,
        expert_ids: List[int]
    ) -> torch.Tensor:
        """Compose multiple experts into a synthetic expert."""
        if len(expert_ids) == 1:
            return self.expert_embeddings[expert_ids[0]]
        
        # Pairwise composition
        composed = self.expert_embeddings[expert_ids[0]]
        for i in range(1, len(expert_ids)):
            pair = torch.cat([composed, self.expert_embeddings[expert_ids[i]]])
            composed = self.composer(pair)
        
        return composed
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        base_expert_outputs: List[torch.Tensor],
        composition_weights: torch.Tensor
    ) -> torch.Tensor:
        """
        Compose outputs using learned composition.
        
        Args:
            hidden_states: Input (B, S, D)
            base_expert_outputs: List of expert outputs
            composition_weights: Which compositions to use
        """
        # Stack base outputs
        stacked = torch.stack(base_expert_outputs, dim=2)  # (B, S, E, D)
        
        # Compute composition coefficients from embeddings
        composed_embedding = torch.einsum(
            'e c, b s e -> b s c',
            self.expert_embeddings,
            composition_weights
        )
        
        # Decode to mixing weights
        mix_weights = F.softmax(self.decoder(composed_embedding), dim=-1)  # (B, S, E)
        
        # Compose outputs
        output = torch.einsum('b s e d, b s e -> b s d', stacked, mix_weights)
        
        return output


# =============================================================================
# 4. MEMORY EXPERTS - Experts with persistent KV cache
# =============================================================================

class MemoryExpert(nn.Module):
    """
    Expert with persistent key-value memory.
    
    Unlike regular experts that are stateless, memory experts
    remember important information across forward passes.
    """
    
    def __init__(
        self,
        hidden_size: int,
        memory_size: int = 128,
        num_heads: int = 4
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.memory_size = memory_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        
        # Persistent memory
        self.memory_keys = nn.Parameter(torch.randn(memory_size, hidden_size))
        self.memory_values = nn.Parameter(torch.randn(memory_size, hidden_size))
        
        # Query projection
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        
        # Memory update gate
        self.update_gate = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input (B, S, D)
        """
        B, S, D = x.shape
        
        # Query against memory
        q = self.q_proj(x)  # (B, S, D)
        
        # Attention over memory
        attn_weights = torch.einsum('bsd, md -> bsm', q, self.memory_keys)
        attn_weights = attn_weights / math.sqrt(self.head_dim)
        attn_weights = F.softmax(attn_weights, dim=-1)
        
        # Retrieve from memory
        retrieved = torch.einsum('bsm, md -> bsd', attn_weights, self.memory_values)
        
        # Combine with input
        output = self.out_proj(retrieved)
        
        # Update memory based on new input (running average)
        if self.training:
            with torch.no_grad():
                # Find most similar memory slots
                similarity = torch.einsum('bsd, md -> bsm', x, self.memory_keys)
                top_slots = similarity.mean(dim=(0,1)).topk(min(16, self.memory_size)).indices
                
                # Soft update
                gate = self.update_gate(
                    torch.cat([x.mean(dim=(0,1)), self.memory_values[top_slots].mean(0)], dim=-1)
                )
                for slot in top_slots:
                    self.memory_values.data[slot] = (
                        gate * x.mean(dim=(0,1)) +
                        (1 - gate) * self.memory_values.data[slot]
                    )
        
        return output


# =============================================================================
# 5. RECURSIVE EXPERTS - Self-calling for hard problems
# =============================================================================

class RecursiveExpert(nn.Module):
    """
    Expert that can call itself recursively for hard problems.
    
    Inspired by test-time compute scaling (o1-style).
    Easy problems: 1 forward pass
    Hard problems: multiple recursive refinements
    """
    
    def __init__(
        self,
        hidden_size: int,
        max_depth: int = 3
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.max_depth = max_depth
        
        # Core computation
        self.compute = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Linear(hidden_size * 4, hidden_size)
        )
        
        # Difficulty estimator (should we recurse?)
        self.difficulty_gate = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 4),
            nn.GELU(),
            nn.Linear(hidden_size // 4, 1),
            nn.Sigmoid()
        )
        
        # Residual combiner
        self.combiner = nn.Linear(hidden_size * 2, hidden_size)
    
    def forward(
        self,
        x: torch.Tensor,
        depth: int = 0
    ) -> Tuple[torch.Tensor, int]:
        """
        Args:
            x: Input (B, S, D)
            depth: Current recursion depth
            
        Returns:
            output: Processed output
            actual_depth: How deep we went
        """
        # Base computation
        computed = self.compute(x)
        
        if depth >= self.max_depth:
            return computed, depth
        
        # Check if we need to recurse
        difficulty = self.difficulty_gate(computed).mean()
        
        if difficulty > 0.5:  # Hard problem
            # Recurse
            refined, final_depth = self.forward(computed, depth + 1)
            
            # Combine recursive result with current
            combined = torch.cat([computed, refined], dim=-1)
            output = self.combiner(combined)
            
            return output, final_depth
        else:
            return computed, depth


# =============================================================================
# COMBINED: SRDE With All Innovations
# =============================================================================

class InnovativeSRDELayer(nn.Module):
    """
    SRDE layer with all experimental innovations combined.
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_experts: int = 8,
        max_seq_len: int = 4096
    ):
        super().__init__()
        
        # Choose router type
        self.temporal_router = TemporalRouter(hidden_size, num_experts)
        self.uncertainty_router = UncertaintyRouter(hidden_size, num_experts)
        
        # Memory-augmented experts
        self.memory_experts = nn.ModuleList([
            MemoryExpert(hidden_size, memory_size=64)
            for _ in range(num_experts)
        ])
        
        # Recursive refinement for hard tokens
        self.recursive_expert = RecursiveExpert(hidden_size)
        
        # Expert composition
        self.composer = ComposableExperts(hidden_size, num_experts)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Full innovative forward pass."""
        B, S, D = hidden_states.shape
        
        if positions is None:
            positions = torch.arange(S, device=hidden_states.device)
        
        # 1. Temporal routing
        temporal_weights, temporal_indices = self.temporal_router(hidden_states, positions)
        
        # 2. Uncertainty routing
        uncertainty_weights, uncertainty_indices, k_values = self.uncertainty_router(hidden_states)
        
        # 3. Process through memory experts
        expert_outputs = []
        for i, expert in enumerate(self.memory_experts):
            out = expert(hidden_states)
            expert_outputs.append(out)
        
        # 4. Compose experts
        avg_weights = (temporal_weights + uncertainty_weights) / 2
        composed_output = self.composer(
            hidden_states,
            expert_outputs,
            avg_weights
        )
        
        # 5. Recursive refinement for high-uncertainty tokens
        high_uncertainty = (k_values > 2).any(dim=-1) if k_values.dim() > 1 else k_values > 2
        if high_uncertainty.any():
            refined, depth = self.recursive_expert(composed_output)
            composed_output = torch.where(
                high_uncertainty.unsqueeze(-1).unsqueeze(-1).expand_as(composed_output),
                refined,
                composed_output
            )
        
        return composed_output


print("SRDE v2.2 Experimental Innovations loaded!")
print("New features: TemporalRouter, UncertaintyRouter, ComposableExperts,")
print("              MemoryExpert, RecursiveExpert")
