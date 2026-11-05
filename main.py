import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization"""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # RMS = sqrt(mean(x^2))
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        x_normed = x / rms
        return self.weight * x_normed


class RotaryPositionEmbedding(nn.Module):
    """Rotary Position Embeddings (RoPE)"""

    def __init__(self, dim: int, max_seq_len: int = 2048, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base

        # Precompute frequencies
        inv_freq = 1.0 / (self.base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

        # Precompute cos and sin for max sequence length
        t = torch.arange(max_seq_len, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos())
        self.register_buffer("sin_cached", emb.sin())

    def rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        """Rotates half the hidden dims of the input."""
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2:]
        return torch.cat((-x2, x1), dim=-1)

    def forward(self, q: torch.Tensor, k: torch.Tensor, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply rotary embeddings to queries and keys"""
        # Shape (1, seq, 1, head_dim) so it broadcasts over batch and heads.
        cos = self.cos_cached[:seq_len, ...][None, :, None, :]
        sin = self.sin_cached[:seq_len, ...][None, :, None, :]

        # Apply rotation
        q_embed = (q * cos) + (self.rotate_half(q) * sin)
        k_embed = (k * cos) + (self.rotate_half(k) * sin)

        return q_embed, k_embed


class GroupedQueryAttention(nn.Module):
    """Grouped Query Attention (GQA)"""

    def __init__(
        self,
        dim: int,
        n_heads: int,
        n_kv_heads: int,
        max_seq_len: int = 2048,
        dropout: float = 0.0
    ):
        super().__init__()
        assert dim % n_heads == 0, "dim must be divisible by n_heads"
        assert n_heads % n_kv_heads == 0, "n_heads must be divisible by n_kv_heads"

        self.dim = dim
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.n_rep = n_heads // n_kv_heads  # Number of query heads per kv head
        self.head_dim = dim // n_heads

        # Projections
        self.wq = nn.Linear(dim, n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(dim, n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(dim, n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(n_heads * self.head_dim, dim, bias=False)

        self.dropout = nn.Dropout(dropout)
        self.rope = RotaryPositionEmbedding(self.head_dim, max_seq_len)

    def repeat_kv(self, x: torch.Tensor) -> torch.Tensor:
        """Repeat k/v heads to match number of query heads"""
        batch_size, seq_len, n_kv_heads, head_dim = x.shape
        if self.n_rep == 1:
            return x
        return x[:, :, :, None, :].expand(batch_size, seq_len, n_kv_heads, self.n_rep, head_dim).reshape(
            batch_size, seq_len, n_kv_heads * self.n_rep, head_dim
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape

        # Project to Q, K, V
        q = self.wq(x)
        k = self.wk(x)
        v = self.wv(x)

        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.n_heads, self.head_dim)
        k = k.view(batch_size, seq_len, self.n_kv_heads, self.head_dim)
        v = v.view(batch_size, seq_len, self.n_kv_heads, self.head_dim)

        # Apply RoPE
        q, k = self.rope(q, k, seq_len)

        # Repeat key/value heads for grouped queries
        k = self.repeat_kv(k)
        v = self.repeat_kv(v)

        # Attention
        attn_scores = torch.einsum(
            "bqhd,bkhd->bhqk", q, k) / math.sqrt(self.head_dim)
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, float("-inf"))
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)

        out = torch.einsum("bhqk,bkhd->bqhd", attn_probs, v)
        out = out.reshape(batch_size, seq_len, self.n_heads * self.head_dim)

        return self.wo(out)


class SwiGLU(nn.Module):
    """SwiGLU activation"""

    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(dim, hidden_dim, bias=False)
        self.w3 = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w3(F.silu(self.w1(x)) * self.w2(x))


class TransformerBlock(nn.Module):
    """Transformer block with GQA, RoPE, SwiGLU, RMSNorm"""

    def __init__(
        self,
        dim: int,
        n_heads: int,
        n_kv_heads: int,
        max_seq_len: int = 2048,
        ff_mult: int = 4,
        dropout: float = 0.0
    ):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.attn = GroupedQueryAttention(dim, n_heads, n_kv_heads, max_seq_len, dropout)
        self.norm2 = RMSNorm(dim)
        self.ff = SwiGLU(dim, ff_mult * dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = x + self.dropout(self.attn(self.norm1(x), mask))
        x = x + self.dropout(self.ff(self.norm2(x)))
        return x


class TransformerLM(nn.Module):
    """Full Transformer Language Model with shared input/output embeddings"""

    def __init__(
        self,
        vocab_size: int,
        dim: int,
        n_layers: int,
        n_heads: int,
        n_kv_heads: int,
        max_seq_len: int = 2048,
        ff_mult: int = 4,
        dropout: float = 0.0
    ):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim)
        self.blocks = nn.ModuleList([
            TransformerBlock(dim, n_heads, n_kv_heads, max_seq_len, ff_mult, dropout)
            for _ in range(n_layers)
        ])
        self.norm_final = RMSNorm(dim)

        # Shared embedding weights
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)
        self.lm_head.weight = self.embed.weight

    def forward(self, tokens: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if mask is None:
            # Causal mask: a position may only attend to itself and earlier tokens.
            seq_len = tokens.shape[1]
            mask = torch.tril(torch.ones(seq_len, seq_len, device=tokens.device, dtype=torch.bool))[None, None, :, :]
        x = self.embed(tokens)
        for block in self.blocks:
            x = block(x, mask)
        x = self.norm_final(x)
        logits = self.lm_head(x)  # shared embedding matrix
        return logits


# Example instantiation:
if __name__ == "__main__":
    model = TransformerLM(
        vocab_size=32000,
        dim=512,
        n_layers=6,
        n_heads=8,
        n_kv_heads=4,
        max_seq_len=1024
    )
    tokens = torch.randint(0, 32000, (2, 64))
    out = model(tokens)
    print(out.shape)  # (batch, seq, vocab)
