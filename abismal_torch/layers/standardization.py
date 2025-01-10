from typing import Optional

import torch


class Standardization(torch.nn.Module):
    def __init__(
        self,
        center: Optional[bool] = True,
        decay: float = 0.999,
        epsilon: float = 1e-6,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.decay = decay
        self.center = center
        self.epsilon = epsilon

    def _debiased_mean_variance(self, data):
        # TODO: tfs.moving_mean_variance_zero_debiased
        var, mean = torch.var_mean(data, dim=0)
        return mean, var

    @property
    def mean(self):
        mean, _ = self._debiased_mean_variance()
        return mean

    @property
    def var(self):
        _, var = self._debiased_mean_variance()
        return var

    @property
    def std(self):
        s = torch.sqrt(self.var)
        return torch.clamp(s, self.epsilon, torch.inf)

    def forward(self, data):
        mean, var = self._debiased_mean_variance(data)
        std = torch.clamp(torch.sqrt(var), self.epsilon, torch.inf)
        if self.center:
            return (data - mean) / std
        return data / std
