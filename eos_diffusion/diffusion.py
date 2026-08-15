"""
v-prediction DDPM: Diffusion Process, Conditoning and Loss for EOS Reconstruction

Wraps the noise schedule, forward process, conditioning, and loss.

Forward process (adding noise):
    x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1-alpha_bar_t) * epsilon

v-prediction:
    v_t = sqrt(alpha_bar_t) * epsilon - sqrt(1-alpha_bar_t) * x_0

Main conditioning via random masking during training:
    50% random masks: 5-30 known points everywhere on the grid
    30% clustered masks: 5-15 known points in the first 30% of the grid
    20% unconditionial: mask = 0 everywhere (for diversity)
"""

import torch
import torch.nn.functional as F

from .model import EOSDiffusionNet
from .schedule import CosineSchedule

class VPredictionDDPM:
    def __init__(self, model: EOSDiffusionNet, schedule: CosineSchedule, device: torch.device):
        self.model = model
        self.schedule = schedule.to(device)
        self.device = device

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor = None) -> torch.Tensor:
        """
        Sample x_t from q(x_t | x_0) = N(sqrt(alpha_bar_t) * x_0, (1-alpha_bar_t)*I)

        Args:
            x_0 : (B, L) clean normalized EOS
            t   : (B,) timesteps (integers in [1, T])
            noise : (B, L) optional pre-samples noise
        Returns:
            x_t : (B, L) noise verson of x0
        """
        if noise is None:
            noise = torch.randn_like(x0)
        sqrt_bar = self.schedule.sqrt_alpha_bar[t][:, None]
        sqrt_1m_abar = self.schedule.sqrt_one_minus_alpha_bar[t][:, None]
        return sqrt_bar * x0 + sqrt_1m_abar * noise

    def compute_v_target(self, x0: torch.Tensor, noise: torch.Tensor,
                         t: torch.Tensor) -> torch.Tensor:
        """
        v_t = sqrt(alpha_bar_t) * epsilon - sqrt(1-alpha_bar_t) * x_0
        """
        sqrt_bar    = self.schedule.sqrt_alpha_bar[t][:, None]
        sqrt_1m_abar = self.schedule.sqrt_one_minus_alpha_bar[t][:, None]

        return sqrt_bar * noise - sqrt_1m_abar * x0

    def predict_x0_from_v(self, x_t: torch.Tensor, v_pred: torch.Tensor,
                          t: torch.Tensor) -> torch.Tensor:
        """
        x_0 = sqrt(alpha_bar_t) * x_t - sqrt(1-alpha_bar_t) * v
        Used during sampling (and it can also be used for gradient guidance.)
        """
        sqrt_bar    = self.schedule.sqrt_alpha_bar[t][:, None]
        sqrt_1m_abar = self.schedule.sqrt_one_minus_alpha_bar[t][:, None]

        return sqrt_bar * x_t - sqrt_1m_abar * v_pred

    def predict_eps_from_v(self, x_t: torch.Tensor, v_pred: torch.Tensor,
                           t: torch.Tensor) -> torch.Tensor:
        """
        epsilon_tilde = sqrt(1-alpha_bar_t) * x_t + sqrt(alpha_bar_t) * v
        Used during sampling to compute the DDPM update step.
        """
        sqrt_bar    = self.schedule.sqrt_alpha_bar[t][:, None]
        sqrt_1m_abar = self.schedule.sqrt_one_minus_alpha_bar[t][:, None]

        return sqrt_1m_abar * x_t + sqrt_bar * v_pred

    def random_mask(self, x0: torch.Tensor) -> tuple:
        """
        Generate random conditioning masks for training.
        Distribution over mask types:
          50%  random:      5-30 known points uniformly across the grid
          30%  clustered:   5-15 known points in the first 30% of the grid
          20%  unconditional: mask = all zeros
        Args:
            x0 : (B, L) clean data (used for shape and for x_cond values)
        Returns:
            mask   : (B, L) binary, 1 = known
            x_cond : (B, L) clean values where mask=1, zero elsewhere
        """
        B, L = x0.shape
        device = x0.device

        # Decide mask type per sample
        r = torch.rand(B, device=device)
        is_random    = (r < 0.5)                     # 50%
        is_clustered = (r >= 0.5) & (r < 0.8)       # 30%
        # remaining 20%: unconditional (mask stays zero)

        # Initialize mask to zeros
        mask = torch.zeros(B, L, device=device)

        # Random masks: 5-30 known points anywhere on the grid
        idx_r = is_random.nonzero(as_tuple=True)[0]
        if idx_r.numel() > 0:
            for i in idx_r:
                n = torch.randint(5, 31, (1,), device=device).item()
                perm = torch.randperm(L, device=device)[:n]
                mask[i, perm] = 1.0

        # Clustered masks: 5-15 known points in first 30% of grid
        idx_c = is_clustered.nonzero(as_tuple=True)[0]
        if idx_c.numel() > 0:
            low_t_end = max(int(0.3 * L), 15)  # first 30% of grid points
            for i in idx_c:
                n = torch.randint(5, 16, (1,), device=device).item()
                perm = torch.randperm(low_t_end, device=device)[:n]
                mask[i, perm] = 1.0

        # Unconditional (20%): mask stays zero (already initialized)
        #Condition values: clean data where mask=1, zero elsewhere
        x_cond = x0 * mask
        return mask, x_cond

    def training_step(self, x0: torch.Tensor) -> torch.Tensor:
        """
        One training iteration. Returns scalar MSE loss.
        Steps:
          1. Sample random timestep t from Uniform{1, ..., T} for each element
          2. Sample noise epsilon from N(0, I)
          3. Compute x_t via forward process
          4. Generate random conditioning mask
          5. Compute v-target
          6. Network predicts v from (x_t, t, mask, x_cond)
          7. Loss = MSE(v_pred, v_target)  averaged over batch and grid
        """
        B = x0.shape[0]
        t = torch.randint(1, self.schedule.T + 1, (B,), device=self.device)
        noise = torch.randn_like(x0)
        x_t = self.q_sample(x0, t, noise)
        mask, x_cond = self.random_mask(x0)
        v_target = self.compute_v_target(x0, noise, t)
        v_pred = self.model(x_t, t.float(), mask, x_cond)
        loss = F.mse_loss(v_pred, v_target)
        return loss