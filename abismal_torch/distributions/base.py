import torch


class DistributionBase:
    def log_prob(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z (torch.Tensor): A tensor of shape (mc_samples, n_reflections).

        Returns:
            torch.Tensor: A tensor of same shape as z containing the log probabilities.
        """
        raise NotImplementedError("Derived classes must implement log_prob()")
