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
            if no KL divergence is implemented for the posterior-prior pair. Should be of
            shape (mc_samples, rac_size).

    Returns:
        KL divergence (torch.Tensor): A scalar averaged over all dimensions.
    """
    try:
        return torch.distributions.kl.kl_divergence(q, p)
    except NotImplementedError or AttributeError:
        if not torch.isfinite(samples).all() or (samples == 0).any():
            print("Samples are not finite or are exactly 0", flush=True)
            from IPython import embed
            embed(colors="linux")
        q_log_prob = q.log_prob(samples)
        p_log_prob = p.log_prob(samples)
        if not torch.isfinite(q_log_prob).all() or not torch.isfinite(p_log_prob).all():
            print("Log probabilities are not finite", flush=True)
            from IPython import embed
            embed(colors="linux")
        # If independent distribution, there is batch dimension but not event dimension.
        # The log_prob has same shape as samples (mc_samples, rac_size). If distribution
        # is joint, such as low-rank multivariate normal, there is event dimension but
        # not batch dimension. The log_prob has shape (mc_samples, ).
        q_shape = q_log_prob.shape
        p_shape = p_log_prob.shape
        if len(q_shape) > len(p_shape):
            kl_div = q_log_prob.mean(dim=1) - p_log_prob
        elif len(q_shape) < len(p_shape):
            kl_div = q_log_prob - p_log_prob.mean(dim=1)
        else:
            kl_div = q_log_prob - p_log_prob
        return kl_div.mean()