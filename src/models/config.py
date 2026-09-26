"""
Configuration classes for models and training.

Uses Pydantic for validation and YAML serialization support.
"""

from pydantic import BaseModel, ConfigDict, Field


class ModelConfig(BaseModel):
    """Configuration for the GatedTransformer model."""

    model_config = ConfigDict(extra="forbid")

    # Vocabulary and sequence
    vocab_size: int = 50257  # GPT-2 tokenizer
    max_seq_len: int = 512

    # Architecture
    n_layers: int = 6
    d_model: int = 256
    n_heads: int = 4
    d_ff: int = 1024  # Feed-forward hidden dimension
    dropout: float = 0.1

    # Gate configuration
    d_gate: int = 32  # Gate embedding dimension (per head)
    sparsity_mode: str = "topk"  # "topk", "threshold", "soft", "stochastic"
    sparsity_k: int = 64  # For topk mode: keys attended per query
    sparsity_threshold: float = 0.0  # For threshold mode
    gate_temperature: float = 1.0  # For soft mode

    # Optional: per-layer sparsity (list of k values, one per layer)
    per_layer_sparsity: list[int] | None = None

    # Freeze gate projections (for random projection control experiments)
    freeze_gates: bool = False


class TrainingConfig(BaseModel):
    """Configuration for training."""

    model_config = ConfigDict(extra="forbid")

    # Optimization
    batch_size: int = 32
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    max_grad_norm: float = 1.0

    # Schedule
    max_steps: int = 50000
    warmup_steps: int = 1000

    # Loss weights
    lambda_sparse: float = 0.0  # Sparsity regularization weight
    lambda_approx: float = 0.0  # Approximation loss weight (vs full attention)
    target_sparsity: float = 0.5  # Target sparsity ratio for regularization

    # Logging and checkpointing
    log_every: int = 100
    eval_every: int = 1000
    save_every: int = 5000

    # Data loading
    num_workers: int = 4


class DataConfig(BaseModel):
    """Configuration for data paths."""

    model_config = ConfigDict(extra="forbid")

    train_path: str = "wikitext103/train.pt"
    valid_path: str = "wikitext103/validation.pt"
    test_path: str = "wikitext103/test.pt"


class ExperimentConfig(BaseModel):
    """Full experiment configuration combining all configs."""

    model_config = ConfigDict(extra="forbid")

    model: ModelConfig = Field(default_factory=ModelConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    data: DataConfig = Field(default_factory=DataConfig)

    @classmethod
    def from_yaml(cls, path: str) -> "ExperimentConfig":
        """Load config from YAML file."""
        import yaml

        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**data)

    def to_yaml(self, path: str) -> None:
        """Save config to YAML file."""
        import yaml

        with open(path, "w") as f:
            yaml.dump(self.model_dump(), f, default_flow_style=False)
