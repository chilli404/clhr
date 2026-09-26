"""
GatedTransformer model for language modeling with sparse attention.

Uses GatedSparseAttention layers to enable efficient training on long sequences
while learning which token pairs need to attend to each other.
"""

import torch
import torch.nn as nn

from .config import ModelConfig
from .gated_attention import GatedAttentionConfig, GatedSparseAttention


class TransformerBlock(nn.Module):
    """Single transformer block with gated sparse attention."""

    def __init__(self, config: ModelConfig, layer_idx: int):
        super().__init__()

        # Determine sparsity for this layer
        if config.per_layer_sparsity is not None:
            sparsity_k = config.per_layer_sparsity[layer_idx]
        else:
            sparsity_k = config.sparsity_k

        attn_config = GatedAttentionConfig(
            d_model=config.d_model,
            n_heads=config.n_heads,
            d_gate=config.d_gate,
            sparsity_mode=config.sparsity_mode,
            sparsity_k=sparsity_k,
            sparsity_threshold=config.sparsity_threshold,
            temperature=config.gate_temperature,
            dropout=config.dropout,
            freeze_gates=config.freeze_gates,
        )

        self.attention = GatedSparseAttention(attn_config)
        self.attn_norm = nn.LayerNorm(config.d_model)

        self.ff = nn.Sequential(
            nn.Linear(config.d_model, config.d_ff),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_ff, config.d_model),
            nn.Dropout(config.dropout),
        )
        self.ff_norm = nn.LayerNorm(config.d_model)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Forward pass with pre-norm architecture.

        Args:
            x: Input tensor [batch, seq_len, d_model]
            attention_mask: Optional mask [batch, 1, 1, seq_len] or similar

        Returns:
            Output tensor [batch, seq_len, d_model]
        """
        # Pre-norm architecture (more stable for training)
        x = x + self.attention(self.attn_norm(x), attention_mask)
        x = x + self.ff(self.ff_norm(x))
        return x


class GatedTransformer(nn.Module):
    """
    Transformer language model with gated sparse attention.

    This model uses learned gate embeddings to determine which token pairs
    should attend to each other, enabling sparse attention patterns that
    are learned end-to-end with the task.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        # Token and position embeddings
        self.embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_embedding = nn.Embedding(config.max_seq_len, config.d_model)
        self.dropout = nn.Dropout(config.dropout)

        # Transformer layers
        self.layers = nn.ModuleList(
            [TransformerBlock(config, i) for i in range(config.n_layers)]
        )

        # Output
        self.final_norm = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Weight tying (reduces parameters and often improves performance)
        self.lm_head.weight = self.embedding.weight

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights with small values for stable training."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Forward pass for language modeling.

        Args:
            input_ids: Token IDs [batch, seq_len]
            attention_mask: Optional padding mask [batch, seq_len], 1=valid, 0=pad

        Returns:
            logits: [batch, seq_len, vocab_size]
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        # Token + positional embeddings
        positions = torch.arange(seq_len, device=device).unsqueeze(0)  # [1, seq_len]
        x = self.embedding(input_ids) + self.pos_embedding(positions)
        x = self.dropout(x)

        # Create causal mask: each position can only attend to previous positions
        causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, seq_len]

        # Combine with padding mask if provided
        if attention_mask is not None:
            # [batch, seq_len] -> [batch, 1, 1, seq_len]
            padding_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            combined_mask = causal_mask * padding_mask
        else:
            combined_mask = causal_mask

        # Forward through transformer layers
        for layer in self.layers:
            x = layer(x, combined_mask)

        # Final norm and output projection
        x = self.final_norm(x)
        logits = self.lm_head(x)

        return logits

    def get_layer_sparsity(self) -> dict[int, float]:
        """Get sparsity ratio for each layer from last forward pass."""
        return {i: layer.attention.get_sparsity_ratio() for i, layer in enumerate(self.layers)}

    def get_approx_errors(self) -> dict[int, float]:
        """Get approximation error for each layer."""
        return {
            i: layer.attention.get_approx_error()
            for i, layer in enumerate(self.layers)
        }

    def get_all_gate_stats(self) -> dict[int, dict[str, float]]:
        """Get gate statistics for each layer from last forward pass."""
        return {i: layer.attention.get_gate_stats() for i, layer in enumerate(self.layers)}

    def num_parameters(self, only_trainable: bool = True) -> int:
        """Count model parameters."""
        if only_trainable:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())

    @classmethod
    def from_checkpoint(cls, checkpoint: dict) -> "GatedTransformer":
        """Load model from checkpoint dictionary."""
        config = ModelConfig(**checkpoint["config"]["model"])
        model = cls(config)
        model.load_state_dict(checkpoint["model_state_dict"])
        return model
