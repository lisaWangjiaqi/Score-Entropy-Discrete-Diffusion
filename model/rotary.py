import torch
from torch import nn


class Rotary(nn.Module):
  

    def __init__(self, dim, base=10_000):
        super().__init__()

        # """(dim // 2,)"""
        inv_freq = 1.0 / (
            base ** (
                torch.arange(0, dim, 2).float() / dim
            )
        )

        self.register_buffer("inv_freq", inv_freq)

        self.seq_len_cached = None
        self.cos_cached = None
        self.sin_cached = None

    def forward(self, x, seq_dim=1):
  

        seq_len = x.shape[seq_dim]

        # Recompute cache if sequence length or device changes
        if (
            seq_len != self.seq_len_cached
            or self.cos_cached is None
            or self.cos_cached.device != x.device
        ):
            self.seq_len_cached = seq_len

            # """(seq_len,)"""
            t = torch.arange(
                seq_len,
                device=x.device,
                dtype=self.inv_freq.dtype,
            )

            # """(seq_len,) x (head_dim // 2,)
            # -> (seq_len, head_dim // 2)"""
            freqs = torch.outer(
                t,
                self.inv_freq.to(x.device)
            )

            # """(seq_len, head_dim // 2)
            # -> (seq_len, head_dim)"""
            emb = torch.cat(
                (freqs, freqs),
                dim=-1
            )

            # """(seq_len, head_dim)
            # -> (1, seq_len, 3, 1, head_dim)"""
            cos = (
                emb.cos()[None, :, None, None, :]
                .repeat(1, 1, 3, 1, 1)
            )

            sin = (
                emb.sin()[None, :, None, None, :]
                .repeat(1, 1, 3, 1, 1)
            )

            # Q and K use RoPE.
            # V should remain unchanged:
            # V_new = V * 1 + rotate(V) * 0
            cos[:, :, 2, :, :] = 1.0
            sin[:, :, 2, :, :] = 0.0

            self.cos_cached = cos
            self.sin_cached = sin

        return self.cos_cached, self.sin_cached


def rotate_half(x):
  
    half_dim = x.shape[-1] // 2

    x1 = x[..., :half_dim]
    x2 = x[..., half_dim:]

    return torch.cat(
        (-x2, x1),
        dim=-1
    )


def apply_rotary_pos_emb(qkv, cos, sin):


    cos = cos.to(
        device=qkv.device,
        dtype=qkv.dtype
    )

    sin = sin.to(
        device=qkv.device,
        dtype=qkv.dtype
    )

    return (
        qkv * cos
        + rotate_half(qkv) * sin
    )