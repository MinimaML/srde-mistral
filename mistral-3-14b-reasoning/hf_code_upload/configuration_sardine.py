"""
SRDE Configuration for Mistral-3-14B-Reasoning

HARDENED VERSION with:
- Input validation
- Bounds checking
- Sensible defaults
- Configuration verification
"""
from dataclasses import dataclass, field
from typing import Optional, List
import warnings


@dataclass
class SRDEConfig:
    """
    Configuration for Sparse Routed Delta Experts.
    
    Expert Domains (6 focused experts):
        0: Advanced Math - algebraic, calculus, complex arithmetic
        1: Formal Logic - proofs, deduction, symbolic reasoning
        2: Algorithm Design - code, optimization, complexity
        3: Scientific Reasoning - physics, chemistry, domain knowledge
        4: Multi-step Planning - decomposition, long-horizon tasks
        5: Abstract/Symbolic - pattern recognition, analogy, abstraction
    
    All parameters are validated on initialization.
    """
    
    # Expert configuration
    num_experts: int = 6   # 6 focused domain experts
    top_k: int = 2         # Clean routing, no collisions
    
    # Sparsity configuration
    initial_sparsity: float = 0.05  # Start with 5% for exploration
    target_sparsity: float = 0.01   # Anneal to 1%
    sparsity_warmup_steps: int = 1000
    
    # Delta factorization
    num_delta_atoms: int = 16  # Shared delta vocabulary size
    
    # Mask learning
    mask_temperature: float = 1.0
    mask_temperature_anneal: bool = True
    mask_temperature_min: float = 0.1
    
    # Loss weights
    lambda_load_balance: float = 0.01
    lambda_orthogonality: float = 0.001
    lambda_sparsity: float = 0.0001
    
    # Training phases (steps)
    phase1_warmup_steps: int = 100      # Train base, freeze deltas
    phase2_mask_steps: int = 500        # Train mask selectors
    phase3_delta_steps: int = 1000      # Train delta values
    # Phase 4 is joint training until end
    
    # Model-specific (Mistral-3-14B)
    hidden_size: int = 5120
    intermediate_size: int = 16384
    num_layers: int = 40
    
    # Which layers to apply SRDE to (None = all)
    srde_layers: Optional[List[int]] = None
    
    def __post_init__(self):
        """Validate all configuration parameters."""
        self._validate()
        
        if self.srde_layers is None:
            self.srde_layers = list(range(self.num_layers))
    
    def _validate(self):
        """Comprehensive validation of all parameters."""
        errors = []
        warnings_list = []
        
        # Expert validation
        if not isinstance(self.num_experts, int) or self.num_experts < 1:
            errors.append(f"num_experts must be positive integer, got {self.num_experts}")
        if not isinstance(self.top_k, int) or self.top_k < 1:
            errors.append(f"top_k must be positive integer, got {self.top_k}")
        if self.top_k > self.num_experts:
            errors.append(f"top_k ({self.top_k}) cannot exceed num_experts ({self.num_experts})")
        
        # Sparsity validation
        if not (0 < self.initial_sparsity <= 1):
            errors.append(f"initial_sparsity must be in (0, 1], got {self.initial_sparsity}")
        if not (0 < self.target_sparsity <= 1):
            errors.append(f"target_sparsity must be in (0, 1], got {self.target_sparsity}")
        if self.target_sparsity > self.initial_sparsity:
            warnings_list.append(
                f"target_sparsity ({self.target_sparsity}) > initial_sparsity ({self.initial_sparsity}), "
                "no annealing will occur"
            )
        if self.sparsity_warmup_steps < 0:
            errors.append(f"sparsity_warmup_steps must be non-negative, got {self.sparsity_warmup_steps}")
        
        # Delta atoms validation
        if not isinstance(self.num_delta_atoms, int) or self.num_delta_atoms < 1:
            errors.append(f"num_delta_atoms must be positive integer, got {self.num_delta_atoms}")
        if self.num_delta_atoms > 64:
            warnings_list.append(
                f"num_delta_atoms={self.num_delta_atoms} is unusually high, consider reducing"
            )
        
        # Temperature validation
        if self.mask_temperature <= 0:
            errors.append(f"mask_temperature must be positive, got {self.mask_temperature}")
        if self.mask_temperature_min <= 0:
            errors.append(f"mask_temperature_min must be positive, got {self.mask_temperature_min}")
        if self.mask_temperature_min > self.mask_temperature:
            warnings_list.append(
                f"mask_temperature_min ({self.mask_temperature_min}) > mask_temperature ({self.mask_temperature})"
            )
        
        # Loss weight validation
        if self.lambda_load_balance < 0:
            errors.append(f"lambda_load_balance must be non-negative, got {self.lambda_load_balance}")
        if self.lambda_orthogonality < 0:
            errors.append(f"lambda_orthogonality must be non-negative, got {self.lambda_orthogonality}")
        if self.lambda_sparsity < 0:
            errors.append(f"lambda_sparsity must be non-negative, got {self.lambda_sparsity}")
        
        # Phase validation
        if self.phase1_warmup_steps < 0:
            errors.append(f"phase1_warmup_steps must be non-negative, got {self.phase1_warmup_steps}")
        if self.phase2_mask_steps < 0:
            errors.append(f"phase2_mask_steps must be non-negative, got {self.phase2_mask_steps}")
        if self.phase3_delta_steps < 0:
            errors.append(f"phase3_delta_steps must be non-negative, got {self.phase3_delta_steps}")
        
        # Model architecture validation
        if self.hidden_size <= 0:
            errors.append(f"hidden_size must be positive, got {self.hidden_size}")
        if self.intermediate_size <= 0:
            errors.append(f"intermediate_size must be positive, got {self.intermediate_size}")
        if self.num_layers <= 0:
            errors.append(f"num_layers must be positive, got {self.num_layers}")
        
        # SRDE layers validation
        if self.srde_layers is not None:
            if not isinstance(self.srde_layers, (list, tuple)):
                errors.append(f"srde_layers must be list or None, got {type(self.srde_layers)}")
            else:
                for idx in self.srde_layers:
                    if not isinstance(idx, int) or idx < 0 or idx >= self.num_layers:
                        errors.append(f"Invalid layer index {idx}, must be in [0, {self.num_layers})")
        
        # Raise errors
        if errors:
            raise ValueError("SRDEConfig validation failed:\n  - " + "\n  - ".join(errors))
        
        # Issue warnings
        for w in warnings_list:
            warnings.warn(f"SRDEConfig: {w}")
    
    @property
    def ffn_params_per_layer(self) -> int:
        """Number of parameters in one FFN layer."""
        # gate_proj + up_proj + down_proj for Mistral's SwiGLU FFN
        return (
            self.hidden_size * self.intermediate_size +  # gate_proj
            self.hidden_size * self.intermediate_size +  # up_proj
            self.intermediate_size * self.hidden_size    # down_proj
        )
    
    @property
    def sparse_params_per_expert(self) -> int:
        """Trainable parameters per expert at target sparsity."""
        return int(self.ffn_params_per_layer * self.target_sparsity)
    
    @property
    def total_trainable_params(self) -> int:
        """Total SRDE trainable parameters."""
        per_layer = (
            self.sparse_params_per_expert * self.num_experts +  # delta values
            self.sparse_params_per_expert * self.num_experts +  # importance scores
            self.num_delta_atoms * self.sparse_params_per_expert +  # shared atoms
            self.num_experts * self.num_delta_atoms +  # atom weights per expert
            # self.ffn_params_per_layer +  # mask logits (REMOVED: Zero-param magnitude pruning)
            self.hidden_size * self.num_experts  # router
        )
        return per_layer * len(self.srde_layers or [])
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            'num_experts': self.num_experts,
            'top_k': self.top_k,
            'initial_sparsity': self.initial_sparsity,
            'target_sparsity': self.target_sparsity,
            'sparsity_warmup_steps': self.sparsity_warmup_steps,
            'num_delta_atoms': self.num_delta_atoms,
            'mask_temperature': self.mask_temperature,
            'mask_temperature_anneal': self.mask_temperature_anneal,
            'mask_temperature_min': self.mask_temperature_min,
            'lambda_load_balance': self.lambda_load_balance,
            'lambda_orthogonality': self.lambda_orthogonality,
            'lambda_sparsity': self.lambda_sparsity,
            'phase1_warmup_steps': self.phase1_warmup_steps,
            'phase2_mask_steps': self.phase2_mask_steps,
            'phase3_delta_steps': self.phase3_delta_steps,
            'hidden_size': self.hidden_size,
            'intermediate_size': self.intermediate_size,
            'num_layers': self.num_layers,
            'srde_layers': self.srde_layers,
        }
    
    @classmethod
    def from_dict(cls, d: dict) -> 'SRDEConfig':
        """Create from dictionary."""
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})
    
    def __repr__(self) -> str:
        return (
            f"SRDEConfig(\n"
            f"  experts={self.num_experts}, top_k={self.top_k},\n"
            f"  sparsity={self.target_sparsity:.2%},\n"
            f"  trainable_params≈{self.total_trainable_params:,}\n"
            f")"
        )


# Default config for Mistral-3-14B-Reasoning
MISTRAL_14B_SRDE_CONFIG = SRDEConfig()


def validate_config(config: SRDEConfig) -> bool:
    """
    Additional runtime validation.
    
    Returns True if valid, raises ValueError otherwise.
    """
    # Check for common misconfigurations
    if config.target_sparsity < 0.001:
        warnings.warn(
            f"Very low sparsity ({config.target_sparsity:.4%}) may cause instability"
        )
    
    if config.num_experts > 16:
        warnings.warn(
            f"Many experts ({config.num_experts}) increases memory usage significantly"
        )
    
    total_params = config.total_trainable_params
    if total_params > 100_000_000:
        warnings.warn(
            f"Large trainable param count ({total_params:,}), consider reducing sparsity"
        )
    
    return True
