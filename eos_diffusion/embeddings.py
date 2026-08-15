"""
Sinusodial Timestep Embedding

Maps scalar timestept t to a vector o dimension dim using sinusodial
positional encoding.

The first dim/2 components are sin(t * freq_k), the rest cos(t * freq_k),
where freq_k = exxp(-ln(1e4) * k /(dim/2)) for k in [0, dim/2 -1[.

This give the network a smooth, high-bandwidh representation for the
noise level so it can modulate its behaviour across the full range of t.
"""

import math, torch, torch.nn as nn

class SinusoidalEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        half = dim // 2
        freqs = torch.exp(-math.log(10000.0) * torch.arange(half) / half)
        self.register_buffer('freqs', freqs) # (dim/2, )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t : (B, ) float32 timestep values
        Returns:
            (B, dim) sinusodial embedding
        """
        args = t[:, None] * self.freqs[None, :] # (B, dim/2)
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1) # (B, dim)