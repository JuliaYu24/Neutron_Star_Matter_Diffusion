"""
EOSDiffusionNet: ResNet-Attention Hybrid for EOS Reconstruction

Complete denoising network. This file wires together the building blocks
from embeddings.py and blocks.py into the full architecture.

Architecture (no downsampling - constant spatial resolution = grid_size):

Input projection:  Conv1d(3, C, k), 3 channels -> C channels
Output projection: GroupNorm -> SiLU -> Conv1d(C, 1, k)

The 3 input channels are:
    [x_t, mask, x_cond]   stacked along dim=1
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .embeddings import SinusoidalEmbedding
from .blocks import ResBlock1D, SelfAttention1D

class EOSDiffusionNet(nn.Module):
    """
    Default hyperparameters:
        hidden_dim = 256    channels throughout the network
        kernel_size = 7     conv kernel
        n_res_blocks = 12   total residual blocks
        n_groups = 3        number of attention-separated groups
        n_heads = 4         attention heads
        dropout = 0.1
    """
    def __init__(self,
                grid_size = 200,
                hidden_dim = 256,
                kernel_size = 7,
                n_res_blocks = 12,
                n_groups = 3,
                n_heads = 4,
                dropout = 0.1):
        super().__init__()
        self.grid_size = grid_size
        time_emb_dim = hidden_dim * 4

        # Timestep embedding: scalar t -> vector
        self.time_mlp = nn.Sequential(
            SinusoidalEmbedding(hidden_dim),
            nn.Linear(hidden_dim, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),)
        # Input: 3 channels (x_t, mask, x_cond) -> hidden_dim channels
        self.input_proj = nn.Conv1d(3, hidden_dim, kernel_size, padding=kernel_size // 2)

        blocks_per_group = n_res_blocks // n_groups
        self.blocks = nn.ModuleList()
        for g in range(n_groups):
            for _ in range(blocks_per_group):
                self.blocks.append(ResBlock1D(hidden_dim, time_emb_dim, kernel_size, dropout))
            self.blocks.append(SelfAttention1D(hidden_dim, n_heads))
        self.output_norm = nn.GroupNorm(min(32, hidden_dim), hidden_dim)
        self.output_proj = nn.Conv1d(hidden_dim, 1, kernel_size, padding=kernel_size // 2)

    def forward(self, x_noisy: torch.Tensor, t: torch.Tensor,
                mask: torch.Tensor, x_cond: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_noisy : (B, grid_size)  noisy EOS curve at timestep t
            t : (B,)            diffusion timestep
            mask : (B, grid_size)  binary mask (1 = known, 0 = unknown)
            x_cond : (B, grid_size)  clean values at known positions, 0 elsewhere
        Returns:
            v_pred : (B, grid_size)  predicted v-target
        """
        x = torch.stack([x_noisy, mask, x_cond], dim=1)
        h = self.input_proj(x)
        t_emb = self.time_mlp(t)
        for block in self.blocks:
            if isinstance(block, ResBlock1D):
                h = block(h, t_emb)
            else:
                h = block(h)
        h = self.output_norm(h)
        h = F.silu(h)
        h = self.output_proj(h)
        return h.squeeze(1)