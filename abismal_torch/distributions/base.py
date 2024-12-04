import torch


class DistributionBase:
    def log_prob(self, z: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Derived classes must implement log_prob()")
