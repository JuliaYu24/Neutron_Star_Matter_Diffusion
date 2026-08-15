"""
Exponential Moving Average (EMA) of Model weights

EMA is essential for diffusion models: the averaged produce
significantly higher-quality samples than the raw training weights.
Typical decay = 0.9999 (update: show <- decay.shadow + (1-decay).param).

Usage:
    ema = EMA(model)
    # After each optimizer step:
    ema.update(model)
    # For evaluating/sampling:
    ema.apply(model) # load the EMA weights into model
    ...sample...
    ema.restore(model) # restore training weights.
"""
import torch, torch.nn as nn

class EMA:
    """ It maintains an exponential moving average of model parameters."""
    def __init__(self, model: nn.Module, decay: float = 0.9999):
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    @torch.no_grad()
    def update(self, model: nn.Module):
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name].lerp_(param.data, 1.0-self.decay)
    def apply(self, model: nn.Module):
        """Replace model params with EMA params. Call restore() to undo."""
        self.backup = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])
    def restore(self, model: nn.Module):
        """Restore original (non-EMA) params after apply()."""
        for name, param in model.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.backup[name])
        self.backup ={}

    def state_dict(self):
        """Return EMA state for checkpointing."""
        return {'decay': self.decay, 'shadow': self.shadow}

    def load_state_dict(self, state_dict):
        """Load EMA state from checkpoing."""
        self.decay = state_dict['decay']
        self.shadow = state_dict['shadow']