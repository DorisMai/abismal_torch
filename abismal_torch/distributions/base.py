from typing import Optional

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


class PosteriorDistributionBase(DistributionBase):
    def rsample(self, sample_shape: tuple[int, ...] = ()) -> torch.Tensor:
        """
        Sample from the posterior distribution.

        Args:
            sample_shape (tuple[int, ...]): The shape of the sample to draw.

        Returns:
            torch.Tensor: A tensor of shape (sample_shape,).
        """
        raise NotImplementedError("Derived classes must implement rsample()")


def compute_kl_divergence(
    q: DistributionBase | torch.distributions.Distribution,
    p: DistributionBase | torch.distributions.Distribution,
    samples: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Compute the KL divergence between the surrogate posterior and the prior.

    Args:
        q (Distribution): The surrogate posterior.
        p (Distribution): The prior.
        samples (torch.Tensor, optional): Samples from the surrogate posterior. Only used
            if no KL divergence is implemented for the posterior-prior pair.

    Returns:
        KL divergence (torch.Tensor): A tensor of shape (Distribution's batch_shape,).
    """
    try:
        return torch.distributions.kl.kl_divergence(q, p)
    except NotImplementedError or AttributeError:
        # if not torch.isfinite(samples).all() or (samples < 0).any():
        #     from IPython import embed
        #     embed(colors="linux")
        kl_div = q.log_prob(samples) - p.log_prob(samples)
        if not torch.isfinite(kl_div).all():
            from IPython import embed

            embed(colors="linux")
        return kl_div.mean(dim=0)
