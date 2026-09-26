"""
Gated Sparse Attention Module.

Core innovation: Learn auxiliary gate embeddings that determine which token pairs
should attend to each other. The gate computation is cheap (low-dimensional dot product)
while the resulting sparse attention pattern can be learned end-to-end.
"""

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GatedAttentionConfig:
    """Configuration for GatedSparseAttention."""

    d_model: int = 256
    n_heads: int = 4
    d_gate: int = 32
    sparsity_mode: str = "topk"  # "topk", "threshold", "soft", "dense", "stochastic"
    sparsity_k: int = 64  # for topk mode: number of keys to attend per query
    sparsity_threshold: float = 0.0  # for threshold mode
    temperature: float = 1.0  # for soft mode
    dropout: float = 0.1
    freeze_gates: bool = False  # if True, W_gq/W_gk are frozen random projections


class GatedSparseAttention(nn.Module):
    """
    Sparse attention with learned gate embeddings.

    Standard attention computes:
        A = softmax(QK^T / sqrt(d))

    We add gate projections G_q, G_k and compute:
        S_gate = G_q @ G_k^T
        M = sparsify(S_gate)  # binary or soft mask
        A = softmax(QK^T / sqrt(d), mask=M)

    The key insight is that gate dimension d_gate can be much smaller than d_head,
    making the gate computation cheap while still being expressive enough to
    learn meaningful sparsity patterns.
    """

    def __init__(self, config: GatedAttentionConfig):
        super().__init__()
        self.config = config

        self.d_head = config.d_model // config.n_heads
        assert config.d_model % config.n_heads == 0, (
            f"d_model ({config.d_model}) must be divisible by n_heads ({config.n_heads})"
        )

        # Standard Q, K, V projections
        self.W_q = nn.Linear(config.d_model, config.d_model, bias=False)
        self.W_k = nn.Linear(config.d_model, config.d_model, bias=False)
        self.W_v = nn.Linear(config.d_model, config.d_model, bias=False)
        self.W_o = nn.Linear(config.d_model, config.d_model, bias=False)

        # Gate projections (only needed for gate-based sparse modes)
        if config.sparsity_mode not in ("dense", "stochastic"):
            self.W_gq = nn.Linear(config.d_model, config.n_heads * config.d_gate, bias=False)
            self.W_gk = nn.Linear(config.d_model, config.n_heads * config.d_gate, bias=False)

            if config.freeze_gates:
                self.W_gq.weight.requires_grad = False
                self.W_gk.weight.requires_grad = False

        self.dropout = nn.Dropout(config.dropout)

        # For analysis - store last forward pass info (detached)
        self.last_gate_scores: torch.Tensor | None = None
        self.last_mask: torch.Tensor | None = None
        self.last_attention: torch.Tensor | None = None
        self.last_full_attention: torch.Tensor | None = None  # for approx error
        self.last_attn_scores_raw: torch.Tensor | None = None  # pre-masking QK^T
        # Non-detached mask for soft mode sparsity regularization gradient flow
        self.last_mask_live: torch.Tensor | None = None

    def compute_gate_mask(
        self,
        g_q: torch.Tensor,  # [batch, n_heads, seq_len, d_gate]
        g_k: torch.Tensor,  # [batch, n_heads, seq_len, d_gate]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute gate scores and sparse mask.

        Args:
            g_q: Query gate embeddings [batch, n_heads, seq_len, d_gate]
            g_k: Key gate embeddings [batch, n_heads, seq_len, d_gate]

        Returns:
            gate_scores: Raw gate scores [batch, n_heads, seq_len, seq_len]
            mask: Sparse mask [batch, n_heads, seq_len, seq_len]
        """
        # Gate scores: simple dot product, scaled
        # [batch, n_heads, seq_len, seq_len]
        gate_scores = torch.matmul(g_q, g_k.transpose(-2, -1))
        gate_scores = gate_scores / (self.config.d_gate**0.5)

        if self.config.sparsity_mode == "topk":
            # Top-k per query: each query attends to its k highest-scoring keys
            k = min(self.config.sparsity_k, gate_scores.size(-1))
            _, topk_indices = torch.topk(gate_scores, k, dim=-1)

            # Create binary mask
            mask = torch.zeros_like(gate_scores)
            mask.scatter_(-1, topk_indices, 1.0)

        elif self.config.sparsity_mode == "threshold":
            # Hard threshold: attend if score > threshold
            mask = (gate_scores > self.config.sparsity_threshold).float()

        elif self.config.sparsity_mode == "soft":
            # Soft mask via sigmoid with temperature
            mask = torch.sigmoid(gate_scores / self.config.temperature)

        else:
            raise ValueError(f"Unknown sparsity mode: {self.config.sparsity_mode}")

        return gate_scores, mask

    def forward(
        self,
        x: torch.Tensor,  # [batch, seq_len, d_model]
        attention_mask: torch.Tensor | None = None,
        return_attention: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with gated sparse attention.

        Args:
            x: Input tensor [batch, seq_len, d_model]
            attention_mask: Optional mask [batch, 1, 1, seq_len] or [batch, 1, seq_len, seq_len]
                           1 = attend, 0 = don't attend
            return_attention: If True, also return attention weights

        Returns:
            output: [batch, seq_len, d_model]
            attention_weights: [batch, n_heads, seq_len, seq_len] (if return_attention)
        """
        batch_size, seq_len, _ = x.shape

        # Compute Q, K, V
        q = self.W_q(x)  # [batch, seq_len, d_model]
        k = self.W_k(x)
        v = self.W_v(x)

        # Reshape for multi-head: [batch, n_heads, seq_len, d_head]
        n_h = self.config.n_heads
        q = q.view(batch_size, seq_len, n_h, self.d_head).transpose(1, 2)
        k = k.view(batch_size, seq_len, n_h, self.d_head).transpose(1, 2)
        v = v.view(batch_size, seq_len, n_h, self.d_head).transpose(1, 2)

        # Standard attention scores
        attn_scores = torch.matmul(
            q, k.transpose(-2, -1)
        ) / (self.d_head**0.5)

        # Normalize attention mask shape once
        if attention_mask is not None:
            if attention_mask.dim() == 2:
                attention_mask = attention_mask.unsqueeze(
                    1
                ).unsqueeze(2)

        # Store raw QK^T scores before any masking (for alignment diagnostics)
        self.last_attn_scores_raw = attn_scores.detach()

        if self.config.sparsity_mode == "dense":
            # Dense: apply causal mask, no gate masking
            if attention_mask is not None:
                attn_scores = attn_scores.masked_fill(
                    attention_mask == 0, float("-inf")
                )
            self.last_gate_scores = None
            self.last_mask = None
            self.last_full_attention = None
        elif self.config.sparsity_mode == "stochastic":
            # Stochastic sparse: fresh random top-k mask each forward pass.
            # No gate projections — the mask is pure random noise.
            # Pre-softmax masking (set to -inf) like topk mode.
            k = min(self.config.sparsity_k, seq_len)

            # Sample random scores and take top-k to get a random k-subset per query
            rand_scores = torch.rand(
                batch_size, n_h, seq_len, seq_len,
                device=x.device,
            )
            _, topk_indices = torch.topk(rand_scores, k, dim=-1)
            gate_mask = torch.zeros_like(rand_scores)
            gate_mask.scatter_(-1, topk_indices, 1.0)

            # Store for analysis
            self.last_gate_scores = None
            self.last_mask = gate_mask.detach()
            self.last_mask_live = None
            self.last_full_attention = None

            # Apply causal mask first, then stochastic mask
            if attention_mask is not None:
                attn_scores = attn_scores.masked_fill(
                    attention_mask == 0, float("-inf")
                )
            attn_scores = attn_scores.masked_fill(
                gate_mask == 0, float("-inf")
            )
        else:
            # Gate-based sparse modes: topk, threshold, soft
            # Compute gate embeddings
            d_g = self.config.d_gate
            g_q = self.W_gq(x)
            g_k = self.W_gk(x)
            g_q = g_q.view(
                batch_size, seq_len, n_h, d_g
            ).transpose(1, 2)
            g_k = g_k.view(
                batch_size, seq_len, n_h, d_g
            ).transpose(1, 2)

            # Compute gate mask
            gate_scores, gate_mask = self.compute_gate_mask(
                g_q, g_k
            )

            # Store for analysis
            self.last_gate_scores = gate_scores.detach()
            self.last_mask = gate_mask.detach()
            # Keep live ref for soft mode regularization
            if self.config.sparsity_mode == "soft":
                self.last_mask_live = gate_mask
            else:
                self.last_mask_live = None

            # Full attention for approx error (eval only)
            if not self.training:
                causal_scores = attn_scores
                if attention_mask is not None:
                    causal_scores = attn_scores.masked_fill(
                        attention_mask == 0, float("-inf")
                    )
                full_attn = F.softmax(causal_scores, dim=-1)
                full_attn = torch.nan_to_num(
                    full_attn, nan=0.0
                )
                self.last_full_attention = (
                    full_attn.detach()
                )
            else:
                self.last_full_attention = None

            # Apply gate mask + causal mask
            if self.config.sparsity_mode in [
                "topk", "threshold"
            ]:
                # Hard: causal then gate (order irrelevant)
                if attention_mask is not None:
                    attn_scores = attn_scores.masked_fill(
                        attention_mask == 0, float("-inf")
                    )
                attn_scores = attn_scores.masked_fill(
                    gate_mask == 0, float("-inf")
                )
            else:
                # Soft: gate THEN causal to avoid
                # -inf * sigmoid NaN in backward
                attn_scores = attn_scores * gate_mask
                if attention_mask is not None:
                    attn_scores = attn_scores.masked_fill(
                        attention_mask == 0, float("-inf")
                    )

        # Softmax and dropout
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = torch.nan_to_num(attn_weights, nan=0.0)
        attn_weights = self.dropout(attn_weights)

        self.last_attention = attn_weights.detach()

        # Apply attention to values
        output = torch.matmul(attn_weights, v)

        # Reshape back: [batch, seq_len, d_model]
        output = (
            output.transpose(1, 2).contiguous()
            .view(batch_size, seq_len, -1)
        )
        output = self.W_o(output)

        if return_attention:
            return output, attn_weights
        return output

    def get_sparsity_ratio(self) -> float:
        """Return actual sparsity ratio from last forward pass."""
        if self.last_mask is None:
            return 0.0

        total = self.last_mask.numel()
        nonzero = self.last_mask.sum().item()
        return 1.0 - (nonzero / total)

    def get_approx_error(self) -> float:
        """Compute attention mass dropped by the sparse mask.

        Measures how much of the full attention distribution is
        discarded by the gate mask. Returns 0 for dense mode.
        """
        if (
            self.last_full_attention is None
            or self.last_mask is None
        ):
            return 0.0

        # Attention mass on positions the mask zeros out
        dropped = (
            self.last_full_attention * (1.0 - self.last_mask)
        )
        return dropped.sum(dim=-1).mean().item()

    def get_gate_stats(self) -> dict[str, float]:
        """Get statistics about the gate embeddings from last forward pass."""
        if self.last_gate_scores is None:
            return {}

        scores = self.last_gate_scores
        return {
            "gate_mean": scores.mean().item(),
            "gate_std": scores.std().item(),
            "gate_min": scores.min().item(),
            "gate_max": scores.max().item(),
            "sparsity_ratio": self.get_sparsity_ratio(),
        }
