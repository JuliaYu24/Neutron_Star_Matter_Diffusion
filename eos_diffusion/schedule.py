"""
Here we define: alpha_bar_r = f(t)/f(0) with f(t) = cos^2((t/T + s)/(1+s) * pi/2)

Key properties:
   - alpha_bar_0 = 1 (no noise at t = 0, i.e. pure signal)
   - alpha_bar_T = 0 (purve noise at t = T)
   - Offset s = 0.008
"""

import math
import torch

class CosineSchedule:
    """
    Precomputes all diffusion schedule tensors:
    alpha_bar[t]               cumulative signal retention (T+1,)
    sqrt_alpha_bar[t]                                      (T+1,)
    sqrt_one_minus_alpha_bar[t]                            (T+1,)
    alpha[t] = alpha_bar[t]/alpha_bar[t-1]                 (T,)
    beta[t] = 1 - alpha[t]                                 (T,)
    posterior_variance[t]       for DDPM sampling step     (T,)
    """
    def __init__(self, T: int = 1000, s: float = 0.008):
        self.T = T
        steps = torch.arange(T + 1, dtype=torch.float64)
        f = torch.cos((steps / T + s) / (1 + s) * math.pi / 2) ** 2
        alpha_bar = f / f[0]
        alpha_bar = torch.clamp(alpha_bar, 1e-5, 0.9999)
        sqrt_alpha_bar = alpha_bar.sqrt() 
        sqrt_1m_abar = (1.0 - alpha_bar).sqrt()
        self.alpha_bar = alpha_bar.float()
        self.sqrt_alpha_bar  = sqrt_alpha_bar.float()
        self.sqrt_one_minus_alpha_bar = sqrt_1m_abar.float()
        alpha = alpha_bar[1:] / alpha_bar[:-1]
        alpha = torch.clamp(alpha, 1e-5, 1.0)
        beta  = 1.0 - alpha
        self.alpha = alpha.float()
        self.beta  = beta.float()
        alpha_bar_prev = alpha_bar[:-1]
        alpha_bar_curr = alpha_bar[1:]
        posterior_var = beta * (1.0 - alpha_bar_prev) / (1.0 - alpha_bar_curr)
        self.posterior_variance = torch.clamp(posterior_var, min=1e-20).float()
    def to(self, device):
        """Move all schedule tensors to the specified device."""
        for name in ['alpha_bar', 'sqrt_alpha_bar', 'sqrt_one_minus_alpha_bar',
                      'alpha', 'beta', 'posterior_variance']:
            setattr(self, name, getattr(self, name).to(device))
        return self