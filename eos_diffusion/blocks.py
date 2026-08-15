"""
Reusable Building Blocks: ResBlock1D + SelfAttention1D

These are the two core building blocks composed by EOSDiffusionNet.
They can also be reused independently for other architectures.
"""
import math, torch, torch.nn as nn, torch.nn.functional as F

# --------------------------------------------------------
# Residual Block (1D convolution + FiLM time conditioning)
# --------------------------------------------------------

class ResBlock1D(nn.Module):

    def __init__(self, channels: int, time_emb_dim: int, kernel_size: int = 7, dropout: float = 0.1):
        super().__init__()
        pad = kernel_size // 2
        self.norm1 = nn.GroupNorm(num_groups = min(32, channels), num_channels=channels)
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=pad)
        self.norm2 = nn.GroupNorm(num_groups = min(32, channels), num_channels=channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=pad)
        self.dropout = nn.Dropout(dropout)

        # FiLM projection: time_emb -> (scale, shift) for 'channel' features
        self.time_proj = nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, 2 * channels),)

        # Zero-init last conv so block starts as identity
        nn.init.zeros_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x     : (B, C, L) feature map
            t_emb : (B, time_emb_dim) timstep embedding
        Returns:
            (B, C, L) residual-updated feature map
        """
        h = self.norm1(x)
        h = F.silu(h)
        h = self.conv1(h)
        # Fil conditioning from timestep
        t = self.time_proj(t_emb)
        scale, shift = t.chunk(2, dim=1)
        h = h * (1.0 + scale[:, :, None]) + shift[:, :, None]
        h = self.norm2(h)
        h = F.silu(h)
        h = self.dropout(h)
        h = self.conv2(h)
        return x + h

class SelfAttention1D(nn.Module):
    """
    Multi-head self-attention over the spatial (grid-point) dimension.

    This gives the network a global receptive field: grid point j = 0
    can directly attend to j = 199. Critical for EOS reconstruction
    because physics imposes long-range correlations across the grid
    pre-nrom (GroupNorm before attention) + zero-initialized output
    proection for residual stability.

    For L = 200 grid points and 4 heads, the attention is only
    200 x 200, which is negligible compute coompared to the 
    convolutional blocks
    """
    def __init__(self, channels: int, n_heads: int = 4):
        super().__init__()
        assert channels % n_heads == 0, f'channels = {channels} not divisible by n_heads={n_heads}'
        self.n_heads = n_heads
        self.head_dim = channels // n_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)

        self.norm = nn.GroupNorm(num_groups=min(32, channels), num_channels=channels)
        self.qkv = nn.Conv1d(channels, 3 * channels, kernel_size = 1)
        self.out = nn.Conv1d(channels, channels, kernel_size=1)

        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : (B, C, L)
        Returns:
            (B, C, L) with global information mixed across all grid points
        """
        B, C, L = x.shape
        h = self.norm(x)
        qkv = self.qkv(h)
        q, k, v = qkv.chunk(3, dim=1)

        # Reshape for multi-head: (B, heads, L, head_dim)
        q = q.view(B, self.n_heads, self.head_dim, L).transpose(2, 3)
        k = k.view(B, self.n_heads, self.head_dim, L).transpose(2, 3)
        v = v.view(B, self.n_heads, self.head_dim, L).transpose(2, 3)

        # scaled dot-product attention
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)

        # reshape back: (B, C, L)
        out = out.transpose(2, 3).contiguous().view(B, C, L)
        out = self.out(out)

        return x + out