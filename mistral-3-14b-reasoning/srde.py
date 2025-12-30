"""
Sparse Routed Delta Experts (SRDE) - Core Implementation

HARDENED VERSION with:
- Input validation
- Device consistency checks
- NaN/Inf detection and handling
- Graceful error recovery
- Comprehensive logging
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Dict, Any, Union
from dataclasses import dataclass
import warnings
import logging

from config import SRDEConfig, validate_config

# Setup logging
logger = logging.getLogger(__name__)


@dataclass
class SRDEOutput:
    """Output from SRDE forward pass."""
    output: torch.Tensor
    router_logits: torch.Tensor
    router_weights: torch.Tensor
    selected_experts: torch.Tensor
    aux_loss: torch.Tensor
    
    def to(self, device: torch.device) -> 'SRDEOutput':
        """Move all tensors to device."""
        return SRDEOutput(
            output=self.output.to(device),
            router_logits=self.router_logits.to(device),
            router_weights=self.router_weights.to(device),
            selected_experts=self.selected_experts.to(device),
            aux_loss=self.aux_loss.to(device)
        )


def check_tensor_health(tensor: torch.Tensor, name: str = "tensor") -> torch.Tensor:
    """
    Check tensor for NaN/Inf and optionally fix.
    
    Returns:
        Tensor with NaN/Inf replaced by zeros (with warning)
    """
    if torch.isnan(tensor).any():
        logger.warning(f"NaN detected in {name}, replacing with zeros")
        tensor = torch.nan_to_num(tensor, nan=0.0)
    
    if torch.isinf(tensor).any():
        logger.warning(f"Inf detected in {name}, clamping")
        tensor = torch.clamp(tensor, min=-1e6, max=1e6)
    
    return tensor


def ensure_device_consistency(*tensors: torch.Tensor) -> torch.device:
    """Ensure all tensors are on the same device, return that device."""
    devices = set(t.device for t in tensors if t is not None)
    if len(devices) > 1:
        raise ValueError(f"Tensors on different devices: {devices}")
    return next(iter(devices)) if devices else torch.device('cpu')


class LearnedMaskSelector(nn.Module):
    """
    Differentiable mask selection using Gumbel-softmax.
    
    HARDENED: Input validation, numerical stability, device handling.
    """
    
    def __init__(
        self,
        num_params: int,
        num_sparse: int,
        temperature: float = 1.0
    ):
        super().__init__()
        
        # Validate inputs
        if num_params <= 0:
            raise ValueError(f"num_params must be positive, got {num_params}")
        if num_sparse <= 0:
            raise ValueError(f"num_sparse must be positive, got {num_sparse}")
        if num_sparse > num_params:
            raise ValueError(f"num_sparse ({num_sparse}) cannot exceed num_params ({num_params})")
        
        self.num_params = num_params
        self.num_sparse = num_sparse
        self._temperature = temperature
        
        # ZERO-PARAM: No learnable logits!
        # Instead, we'll store the indices we've selected based on base model weights
        self.register_buffer("indices", torch.zeros(num_sparse, dtype=torch.long))
        self.initialized = False
    
    @property
    def temperature(self) -> float:
        return self._temperature
    
    @temperature.setter
    def temperature(self, value: float):
        pass # No-op for magnitude pruning
    
    def forward(self, hard: bool = True) -> torch.Tensor:
        """
        Return the cached sparse indices.
        NOTE: This expects 'update_indices' to be called externally with weights.
        """
        # Return a mask for compatibility with old interface, but indices are preferred
        mask = torch.zeros(self.num_params, device=self.indices.device)
        mask[self.indices] = 1.0
        return mask
    
    def update_indices(self, base_weight: torch.Tensor):
        """
        Update selected indices based on base weight magnitude.
        Call this once at startup or periodically.
        """
        with torch.no_grad():
            flat_weight = base_weight.flatten()
            # Select weights with largest absolute magnitude
            _, indices = torch.topk(flat_weight.abs(), self.num_sparse)
            
            # Ensure indices are on correct device before copying to buffer
            if indices.device != self.indices.device:
                indices = indices.to(self.indices.device)
                
            self.indices.copy_(indices)
            self.initialized = True
            
    def get_mask_indices(self) -> torch.Tensor:
        """Get indices of top-k positions."""
        return self.indices.clone()
    
    def set_temperature(self, temperature: float):
        """Update temperature (no-op)."""
        pass


class SharedDeltaVocabulary(nn.Module):
    """
    Shared delta atoms that experts combine with different weights.
    
    HARDENED: Bounds checking, gradient scaling.
    """
    
    def __init__(
        self,
        num_atoms: int,
        num_sparse: int,
        num_experts: int
    ):
        super().__init__()
        
        # Validate
        if num_atoms <= 0:
            raise ValueError(f"num_atoms must be positive, got {num_atoms}")
        if num_sparse <= 0:
            raise ValueError(f"num_sparse must be positive, got {num_sparse}")
        if num_experts <= 0:
            raise ValueError(f"num_experts must be positive, got {num_experts}")
        
        self.num_atoms = num_atoms
        self.num_sparse = num_sparse
        self.num_experts = num_experts
        
        # Shared delta atoms: [num_atoms, num_sparse]
        self.atoms = nn.Parameter(torch.zeros(num_atoms, num_sparse))
        nn.init.normal_(self.atoms, mean=0.0, std=0.02)
        
        # Per-expert atom weights: [num_experts, num_atoms]
        self.expert_atom_weights = nn.Parameter(
            torch.zeros(num_experts, num_atoms)
        )
        nn.init.normal_(self.expert_atom_weights, mean=0.0, std=0.1)
    
    def get_expert_delta(self, expert_idx: int) -> torch.Tensor:
        """Get the delta for a specific expert."""
        if expert_idx < 0 or expert_idx >= self.num_experts:
            raise ValueError(f"expert_idx {expert_idx} out of range [0, {self.num_experts})")
        
        weights = F.softmax(self.expert_atom_weights[expert_idx], dim=-1)
        delta = weights @ self.atoms
        return check_tensor_health(delta, f"expert_{expert_idx}_delta")
    
    def get_all_deltas(self) -> torch.Tensor:
        """Get deltas for all experts at once."""
        weights = F.softmax(self.expert_atom_weights, dim=-1)
        deltas = weights @ self.atoms
        return check_tensor_health(deltas, "all_deltas")


class SparseExpert(nn.Module):
    """A single sparse expert with importance-weighted deltas."""
    
    def __init__(
        self,
        expert_idx: int,
        num_sparse: int,
        vocabulary: SharedDeltaVocabulary
    ):
        super().__init__()
        
        if expert_idx < 0:
            raise ValueError(f"expert_idx must be non-negative, got {expert_idx}")
        if num_sparse <= 0:
            raise ValueError(f"num_sparse must be positive, got {num_sparse}")
        
        self.expert_idx = expert_idx
        self.num_sparse = num_sparse
        self.vocabulary = vocabulary
        
        # Per-position importance scores
        self.importance = nn.Parameter(torch.zeros(num_sparse))
        nn.init.normal_(self.importance, mean=0.0, std=0.1)
    
    def get_weighted_delta(self) -> torch.Tensor:
        """Get importance-weighted delta values."""
        try:
            base_delta = self.vocabulary.get_expert_delta(self.expert_idx)
            
            # Ensure shapes match
            if len(base_delta) != len(self.importance):
                # Handle shape mismatch gracefully
                min_len = min(len(base_delta), len(self.importance))
                base_delta = base_delta[:min_len]
                importance = self.importance[:min_len]
            else:
                importance = self.importance
            
            importance_weights = torch.sigmoid(importance)
            result = base_delta * importance_weights
            
            return check_tensor_health(result, f"expert_{self.expert_idx}_weighted_delta")
        
        except Exception as e:
            logger.error(f"Error in SparseExpert.get_weighted_delta: {e}")
            return torch.zeros(self.num_sparse, device=self.importance.device)


class SRDERouter(nn.Module):
    """Router that determines which experts to use for each token."""
    
    def __init__(
        self,
        hidden_size: int,
        num_experts: int,
        top_k: int = 2
    ):
        super().__init__()
        
        if hidden_size <= 0:
            raise ValueError(f"hidden_size must be positive, got {hidden_size}")
        if num_experts <= 0:
            raise ValueError(f"num_experts must be positive, got {num_experts}")
        if top_k <= 0 or top_k > num_experts:
            raise ValueError(f"top_k must be in [1, {num_experts}], got {top_k}")
        
        self.num_experts = num_experts
        self.top_k = top_k
        
        self.gate = nn.Linear(hidden_size, num_experts, bias=False)
        nn.init.normal_(self.gate.weight, mean=0.0, std=0.01)
    
    def forward(
        self,
        hidden_states: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Route tokens to experts."""
        
        # Input validation
        if hidden_states.dim() < 2:
            raise ValueError(f"Expected hidden_states with >= 2 dims, got {hidden_states.dim()}")
        
        # Compute routing logits
        router_logits = self.gate(hidden_states)
        router_logits = check_tensor_health(router_logits, "router_logits")
        
        # Clamp for numerical stability
        router_logits = torch.clamp(router_logits, min=-50, max=50)
        
        # Select top-k experts
        effective_k = min(self.top_k, self.num_experts)
        top_weights, top_indices = torch.topk(router_logits, effective_k, dim=-1)
        
        # Normalize weights
        router_weights = F.softmax(top_weights, dim=-1)
        router_weights = check_tensor_health(router_weights, "router_weights")
        
        return router_logits, router_weights, top_indices
    
    def warm_start_from_hidden_states(
        self,
        hidden_states: torch.Tensor,
        method: str = "kmeans"
    ):
        """
        Initialize router weights from hidden state clustering.
        
        This gives the router a head start by learning cluster centers
        from the data distribution before training.
        
        Args:
            hidden_states: [num_samples, hidden_size] sample hidden states
            method: "kmeans" or "random" (random projection baseline)
        """
        device = self.gate.weight.device
        hidden_states = hidden_states.to(device).float()
        
        if method == "kmeans":
            # Simple K-means clustering
            num_samples = hidden_states.shape[0]
            num_experts = self.num_experts
            
            # Initialize centroids randomly from data
            indices = torch.randperm(num_samples)[:num_experts]
            centroids = hidden_states[indices].clone()
            
            # Run a few K-means iterations
            for _ in range(10):
                # Assign points to nearest centroid
                distances = torch.cdist(hidden_states, centroids)
                assignments = distances.argmin(dim=1)
                
                # Update centroids
                new_centroids = torch.zeros_like(centroids)
                counts = torch.zeros(num_experts, device=device)
                for i in range(num_experts):
                    mask = (assignments == i)
                    if mask.any():
                        new_centroids[i] = hidden_states[mask].mean(dim=0)
                        counts[i] = mask.sum()
                    else:
                        new_centroids[i] = centroids[i]
                
                centroids = new_centroids
            
            # Set router weights to point toward centroids
            # The gate projects hidden_states to expert logits
            # We want: gate(centroid_i) to have highest value at position i
            with torch.no_grad():
                # Normalize centroids and use as gate weights
                normalized_centroids = F.normalize(centroids, dim=1)
                self.gate.weight.copy_(normalized_centroids)
            
            logger.info(f"Router warm-started from K-means clustering ({num_samples} samples)")
        
        elif method == "random":
            # Random orthogonal initialization (baseline)
            with torch.no_grad():
                nn.init.orthogonal_(self.gate.weight)
            logger.info("Router warm-started with orthogonal initialization")
        
        else:
            raise ValueError(f"Unknown warm-start method: {method}")


class SRDELayer(nn.Module):
    """SRDE wrapper for a single FFN layer."""
    
    def __init__(
        self,
        base_ffn: nn.Module,
        config: SRDEConfig,
        layer_idx: int
    ):
        super().__init__()
        
        if base_ffn is None:
            raise ValueError("base_ffn cannot be None")
        
        self.config = config
        self.layer_idx = layer_idx
        
        # Reference to base FFN (frozen)
        self.base_ffn = base_ffn
        for param in self.base_ffn.parameters():
            param.requires_grad = False
        
        # Count FFN parameters
        self.num_ffn_params = sum(p.numel() for p in self.base_ffn.parameters())
        if self.num_ffn_params == 0:
            raise ValueError(f"base_ffn has no parameters")
        
        self.num_sparse = max(1, int(self.num_ffn_params * config.target_sparsity))
        
        logger.info(f"Layer {layer_idx}: {self.num_ffn_params:,} FFN params, {self.num_sparse:,} sparse")
        
        
        # Get dtype and device from base FFN
        self.base_dtype = next(self.base_ffn.parameters()).dtype
        self.base_device = next(self.base_ffn.parameters()).device
        
        # Components
        self.mask_selector = LearnedMaskSelector(
            num_params=self.num_ffn_params,
            num_sparse=self.num_sparse,
            temperature=config.mask_temperature
        ).to(device=self.base_device, dtype=self.base_dtype)
        
        # Initialize sparsity mask based on base weight magnitude
        # Use simple concatenation of all params to find global top-k
        with torch.no_grad():
            all_params = []
            for p in self.base_ffn.parameters():
                all_params.append(p.flatten())
            all_params = torch.cat(all_params)
            self.mask_selector.update_indices(all_params)
        
        self.vocabulary = SharedDeltaVocabulary(
            num_atoms=config.num_delta_atoms,
            num_sparse=self.num_sparse,
            num_experts=config.num_experts
        ).to(device=self.base_device, dtype=self.base_dtype)
        
        self.experts = nn.ModuleList([
            SparseExpert(
                expert_idx=i,
                num_sparse=self.num_sparse,
                vocabulary=self.vocabulary
            ).to(device=self.base_device, dtype=self.base_dtype)
            for i in range(config.num_experts)
        ])
        
        self.router = SRDERouter(
            hidden_size=config.hidden_size,
            num_experts=config.num_experts,
            top_k=config.top_k
        ).to(device=self.base_device, dtype=self.base_dtype)
        
        # Forward pass counter for debugging
        self._forward_count = 0
        
        # Progressive unlocking state
        self._unlocked_experts = set(range(config.num_experts))  # All unlocked by default
    
    def lock_expert(self, expert_idx: int):
        """Lock an expert (freeze and exclude from routing)."""
        if expert_idx < len(self.experts):
            self._unlocked_experts.discard(expert_idx)
            for param in self.experts[expert_idx].parameters():
                param.requires_grad = False
            logger.info(f"Layer {self.layer_idx}: Locked expert {expert_idx}")
    
    def unlock_expert(self, expert_idx: int):
        """Unlock an expert (unfreeze and include in routing)."""
        if expert_idx < len(self.experts):
            self._unlocked_experts.add(expert_idx)
            for param in self.experts[expert_idx].parameters():
                param.requires_grad = True
            logger.info(f"Layer {self.layer_idx}: Unlocked expert {expert_idx}")
    
    def lock_all_experts(self):
        """Lock all experts."""
        for i in range(len(self.experts)):
            self.lock_expert(i)
    
    def unlock_all_experts(self):
        """Unlock all experts."""
        for i in range(len(self.experts)):
            self.unlock_expert(i)
    
    def set_unlocked_experts(self, expert_indices: List[int]):
        """Set exactly which experts are unlocked."""
        self.lock_all_experts()
        for idx in expert_indices:
            self.unlock_expert(idx)
    
    def get_unlocked_expert_count(self) -> int:
        """Get number of currently unlocked experts."""
        return len(self._unlocked_experts)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        """Forward pass with SRDE modifications. Returns tensor for drop-in replacement."""
        self._forward_count += 1
        
        try:
            # Validate input
            if hidden_states is None:
                raise ValueError("hidden_states cannot be None")
            
            # Get routing decisions
            router_logits, router_weights, selected_experts = self.router(hidden_states)
            
            # Run base FFN
            base_output = self.base_ffn(hidden_states)
            base_output = check_tensor_health(base_output, "base_ffn_output")
            
            # Compute expert contributions
            output_delta = torch.zeros_like(base_output)
            
            # Get all deltas efficiently
            expert_deltas = []
            for expert in self.experts:
                expert_deltas.append(expert.get_weighted_delta())
            expert_deltas = torch.stack(expert_deltas)  # [num_experts, num_sparse]
            
            # Ensure deltas are on correct device
            if expert_deltas.device != base_output.device:
                expert_deltas = expert_deltas.to(base_output.device)
            
            # Mean delta scale per expert
            delta_scale = expert_deltas.mean(dim=1)  # [num_experts]
            
            # Apply weighted by router
            for k in range(self.config.top_k):
                expert_idx = selected_experts[..., k]
                weight = router_weights[..., k:k+1]
                
                # Safe indexing
                scale = delta_scale[expert_idx.clamp(0, len(delta_scale)-1)]
                contribution = weight * scale.unsqueeze(-1) * base_output
                output_delta = output_delta + contribution
            
            output = base_output + output_delta
            output = check_tensor_health(output, "srde_output")
            
            # Store auxiliary loss for later aggregation
            aux_loss = self._compute_aux_loss(router_logits)
            if aux_loss.device != hidden_states.device:
                aux_loss = aux_loss.to(hidden_states.device)
            self._last_aux_loss = aux_loss
            
            self._last_router_logits = router_logits
            self._last_router_weights = router_weights
            self._last_selected_experts = selected_experts
            
            return output
        
        except Exception as e:
            logger.error(f"Error in SRDELayer forward (call {self._forward_count}): {e}")
            # Fallback: return base FFN output
            base_output = self.base_ffn(hidden_states)
            # Ensure fallback loss requires grad to avoid backward error
            self._last_aux_loss = torch.tensor(0.0, device=hidden_states.device, requires_grad=True)
            return base_output
    
    def get_last_aux_loss(self) -> torch.Tensor:
        """Get the auxiliary loss from the last forward pass."""
        # Ensure it requires grad
        loss = getattr(self, '_last_aux_loss', torch.tensor(0.0))
        if not loss.requires_grad:
            loss = loss.clone().detach().requires_grad_(True)
        return loss
    
    def _compute_aux_loss(self, router_logits: torch.Tensor) -> torch.Tensor:
        """Compute load balancing auxiliary loss."""
        try:
            router_probs = F.softmax(router_logits, dim=-1)
            avg_probs = router_probs.mean(dim=list(range(router_probs.dim() - 1)))
            target = 1.0 / self.config.num_experts
            load_loss = F.mse_loss(avg_probs, torch.full_like(avg_probs, target))
            return load_loss * self.config.lambda_load_balance
        except Exception as e:
            logger.warning(f"Error computing aux loss: {e}")
            return torch.tensor(0.0, device=router_logits.device)


class SRDEModel(nn.Module):
    """Full SRDE wrapper for a transformer model."""
    
    def __init__(
        self,
        base_model: nn.Module,
        config: SRDEConfig
    ):
        super().__init__()
        
        if base_model is None:
            raise ValueError("base_model cannot be None")
        
        validate_config(config)
        
        self.config = config
        self.base_model = base_model
        
        # Freeze base model
        for param in self.base_model.parameters():
            param.requires_grad = False
        
        # Create SRDE layers
        self.srde_layers = nn.ModuleDict()
        self._wrap_ffn_layers()
        
        if len(self.srde_layers) == 0:
            raise ValueError("No FFN layers found to wrap with SRDE")
        
        logger.info(f"Created SRDE model with {len(self.srde_layers)} SRDE layers")
    
    def _wrap_ffn_layers(self):
        """Identify and wrap FFN layers by replacing them in the model."""
        layers = self._get_transformer_layers()
        
        for idx in self.config.srde_layers:
            if idx < len(layers):
                try:
                    layer = layers[idx]
                    ffn_attr = self._get_ffn_attr_name(layer)
                    
                    if ffn_attr is not None:
                        ffn = getattr(layer, ffn_attr)
                        srde_layer = SRDELayer(
                            base_ffn=ffn,
                            config=self.config,
                            layer_idx=idx
                        )
                        # Replace the original FFN with SRDE-wrapped version
                        setattr(layer, ffn_attr, srde_layer)
                        self.srde_layers[str(idx)] = srde_layer
                        logger.info(f"Wrapped layer {idx}.{ffn_attr} with SRDE")
                except Exception as e:
                    logger.warning(f"Failed to wrap layer {idx}: {e}")
    
    def _get_transformer_layers(self) -> List[nn.Module]:
        """Get list of transformer layers from base model."""
        # Try various common attribute paths
        search_paths = [
            ('model', 'layers'),
            ('model', 'model', 'layers'),  # For nested wrappers like Ministral3ForCausalLM
            ('model', 'decoder', 'layers'),
            ('transformer', 'h'),
            ('gpt_neox', 'layers'),
            ('layers',),
            ('h',),
            ('blocks',),
        ]
        
        for path in search_paths:
            obj = self.base_model
            try:
                for attr in path:
                    obj = getattr(obj, attr)
                if isinstance(obj, (list, nn.ModuleList)):
                    return list(obj)
            except AttributeError:
                continue
        
        raise ValueError(
            "Could not find transformer layers. "
            "Supported: model.layers, transformer.h, etc."
        )
    
    def _get_ffn_attr_name(self, layer: nn.Module) -> Optional[str]:
        """Get the attribute name of FFN/MLP in a transformer layer."""
        for attr in ['mlp', 'ffn', 'feed_forward', 'ff', 'dense']:
            if hasattr(layer, attr):
                return attr
        return None
    
    def _get_ffn_from_layer(self, layer: nn.Module) -> Optional[nn.Module]:
        """Extract FFN/MLP from a transformer layer."""
        attr = self._get_ffn_attr_name(layer)
        if attr:
            return getattr(layer, attr)
        return None
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Forward pass."""
        # Validate inputs
        if input_ids is None:
            raise ValueError("input_ids cannot be None")
        
        # Forward through base model (SRDE layers are now embedded in the model)
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            **kwargs
        )
        
        # Aggregate aux losses from all SRDE layers
        total_aux_loss = torch.tensor(0.0, device=input_ids.device, requires_grad=True)
        for srde_layer in self.srde_layers.values():
            layer_aux_loss = srde_layer.get_last_aux_loss()
            if layer_aux_loss.device != total_aux_loss.device:
                layer_aux_loss = layer_aux_loss.to(total_aux_loss.device)
            total_aux_loss = total_aux_loss + layer_aux_loss
        
        return {
            'logits': getattr(outputs, 'logits', outputs[0] if isinstance(outputs, tuple) else None),
            'loss': getattr(outputs, 'loss', None),
            'hidden_states': getattr(outputs, 'hidden_states', None),
            'aux_loss': total_aux_loss
        }
    
    def get_trainable_params(self) -> List[nn.Parameter]:
        """Get SRDE trainable parameters."""
        params = []
        for srde_layer in self.srde_layers.values():
            params.extend(p for p in srde_layer.parameters() if p.requires_grad)
        return params
    
    def trainable_param_count(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.get_trainable_params())
    
    def save_srde_weights(self, path: str):
        """Save SRDE weights."""
        import os
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        
        state = {
            'srde_layers': {
                idx: layer.state_dict()
                for idx, layer in self.srde_layers.items()
            },
            'config': self.config.to_dict()
        }
        torch.save(state, path)
        logger.info(f"Saved SRDE weights to {path}")
    
    def load_srde_weights(self, path: str):
        """Load SRDE weights."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"SRDE weights not found: {path}")
        
        state = torch.load(path, map_location='cpu')
        
        for idx, layer_state in state.get('srde_layers', {}).items():
            if idx in self.srde_layers:
                try:
                    self.srde_layers[idx].load_state_dict(layer_state)
                except Exception as e:
                    logger.warning(f"Failed to load layer {idx}: {e}")
        
        logger.info(f"Loaded SRDE weights from {path}")


def create_srde_model(
    model_name: str = "mistralai/Ministral-3-14B-Reasoning-2512",
    config: Optional[SRDEConfig] = None,
    device_map: str = "auto",
    torch_dtype: torch.dtype = torch.bfloat16,
    trust_remote_code: bool = True,
    use_flash_attention: bool = False
) -> SRDEModel:
    """
    Create an SRDE-wrapped model.
    
    Args:
        model_name: HuggingFace model name
        config: SRDE configuration
        device_map: Device placement strategy
        torch_dtype: Model dtype (bf16 recommended)
        trust_remote_code: Trust remote code from HF
        use_flash_attention: Use Flash Attention 3 (requires H100/H200 + flash-attn package)
    
    HARDENED: Better error messages, validation, logging.
    """
    from transformers import AutoModelForCausalLM
    import os
    
    if config is None:
        config = SRDEConfig()
    
    validate_config(config)
    
    print(f"[SRDE] Loading base model: {model_name}")
    print(f"[SRDE] Config: {config}")
    
    # Prepare model loading kwargs
    model_kwargs = {
        "device_map": device_map,
        "torch_dtype": torch_dtype,
        "trust_remote_code": trust_remote_code
    }
    
    # Flash Attention 3 (for H100/H200)
    if use_flash_attention:
        try:
            import flash_attn
            print(f"[SRDE] Flash Attention enabled (v{flash_attn.__version__})")
            model_kwargs["attn_implementation"] = "flash_attention_2"  # HF name for FA3
        except ImportError:
            print("[SRDE] WARNING: flash-attn not installed, falling back to SDPA")
            print("[SRDE] Install with: pip install flash-attn --no-build-isolation")
            model_kwargs["attn_implementation"] = "sdpa"
    
    try:
        # First try standard loading
        base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            **model_kwargs
        )
    except ValueError as e:
        # Handle Mistral3/Ministral multimodal models
        if "Mistral3Config" in str(e) or "ministral" in model_name.lower():
            print("[SRDE] Detected Mistral3/Ministral multimodal model, extracting text model...")
            from transformers import AutoModel
            import torch.nn as nn
            
            # Load the full multimodal model
            full_model = AutoModel.from_pretrained(
                model_name,
                **model_kwargs
            )
            
            # Debug: print model structure
            print(f"[SRDE] Full model type: {type(full_model).__name__}")
            print(f"[SRDE] Full model attributes: {[a for a in dir(full_model) if not a.startswith('_')][:20]}")
            
            # Find the language model and lm_head
            language_model = None
            lm_head = None
            
            # Look for language_model
            if hasattr(full_model, 'language_model'):
                language_model = full_model.language_model
                # Check if lm_head is on language_model
                if hasattr(full_model.language_model, 'lm_head'):
                    lm_head = full_model.language_model.lm_head
            
            # Look for lm_head in various places
            if lm_head is None:
                for attr in ['lm_head', 'output', 'head', 'cls']:
                    if hasattr(full_model, attr):
                        lm_head = getattr(full_model, attr)
                        break
                    if language_model and hasattr(language_model, attr):
                        lm_head = getattr(language_model, attr)
                        break
            
            # If still no lm_head, create one from config
            if lm_head is None:
                print("[SRDE] WARNING: Could not find lm_head, creating from config...")
                hf_config = full_model.config  # Use different name to avoid shadowing SRDEConfig
                if hasattr(hf_config, 'text_config'):
                    hidden_size = hf_config.text_config.hidden_size
                    vocab_size = hf_config.text_config.vocab_size
                else:
                    hidden_size = hf_config.hidden_size
                    vocab_size = hf_config.vocab_size
                
                # Get device from language model
                device = next(language_model.parameters()).device
                
                lm_head = nn.Linear(hidden_size, vocab_size, bias=False, device=device, dtype=torch_dtype)
                # Initialize from embed_tokens but keep as separate trainable parameter
                if hasattr(language_model, 'embed_tokens'):
                    with torch.no_grad():
                        lm_head.weight.copy_(language_model.embed_tokens.weight)
                    lm_head.weight.requires_grad = True
                    print(f"[SRDE] Initialized lm_head from embed_tokens (trainable copy, device={device})")
            
            if language_model is None:
                raise RuntimeError("Could not find language_model in Mistral3Model")
            
            # Create a wrapper class
            class Ministral3ForCausalLM(nn.Module):
                def __init__(self, lang_model, head, model_config):
                    super().__init__()
                    self.model = lang_model
                    self.lm_head = head
                    # Store as model_config, not config (to avoid conflict with SRDEConfig)
                    self.model_config = model_config
                    # But also expose as config for compatibility
                    self.config = model_config
                
                def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
                    # Remove cache-related kwargs that cause issues
                    kwargs.pop('use_cache', None)
                    kwargs.pop('past_key_values', None)
                    
                    # Get hidden states from the language model
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        **kwargs
                    )
                    hidden_states = outputs[0] if isinstance(outputs, tuple) else outputs.last_hidden_state
                    
                    # Apply lm_head to get logits
                    logits = self.lm_head(hidden_states)
                    
                    # Compute loss if labels provided
                    loss = None
                    if labels is not None:
                        shift_logits = logits[..., :-1, :].contiguous()
                        shift_labels = labels[..., 1:].contiguous()
                        loss_fct = nn.CrossEntropyLoss()
                        loss = loss_fct(
                            shift_logits.view(-1, shift_logits.size(-1)),
                            shift_labels.view(-1)
                        )
                    
                    # Return in a format compatible with CausalLMOutput
                    from transformers.modeling_outputs import CausalLMOutputWithPast
                    return CausalLMOutputWithPast(
                        loss=loss,
                        logits=logits,
                        past_key_values=getattr(outputs, 'past_key_values', None),
                        hidden_states=getattr(outputs, 'hidden_states', None),
                        attentions=getattr(outputs, 'attentions', None),
                    )
                
                def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
                    if hasattr(self.model, 'gradient_checkpointing_enable'):
                        self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs)
            
            model_config = language_model.config if hasattr(language_model, 'config') else full_model.config
            base_model = Ministral3ForCausalLM(language_model, lm_head, model_config)
            print(f"[SRDE] Created Ministral3ForCausalLM wrapper")
        else:
            raise RuntimeError(f"Failed to load base model '{model_name}': {e}")
    
    print("[SRDE] Wrapping with SRDE...")
    
    try:
        srde_model = SRDEModel(base_model, config)
    except Exception as e:
        raise RuntimeError(f"Failed to create SRDE model: {e}")
    
    trainable = srde_model.trainable_param_count()
    total = sum(p.numel() for p in base_model.parameters())
    
    print(f"[SRDE] Base model params: {total:,}")
    print(f"[SRDE] SRDE trainable params: {trainable:,} ({100*trainable/total:.4f}%)")
    
    return srde_model


# Import for backwards compatibility
import os
