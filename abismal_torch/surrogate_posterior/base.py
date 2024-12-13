from typing import Optional, Protocol

import torch
from rs_distributions.modules import DistributionModule

from abismal_torch.distributions import PosteriorDistributionBase
from abismal_torch.symmetry import ReciprocalASUCollection


class PosteriorBase(torch.nn.Module):
    def __init__(
        self,
        rac: ReciprocalASUCollection,
        distribution: DistributionModule | PosteriorDistributionBase,
        **kwargs
    ):
        """
        rac : ReciprocalASUCollection
        """
        super().__init__(**kwargs)
        self.rac = rac
        self.distribution = distribution

    def rsample(self, *args, **kwargs) -> torch.Tensor:
        return self.distribution.rsample(*args, **kwargs)

    def log_prob(self, z: torch.Tensor) -> torch.Tensor:
        return self.distribution.log_prob(z)

    # def compute_kl_terms(self, prior, samples=None):
    #     try:
    #         kl_div = self.distribution().kl_divergence(prior)
    #     except NotImplementedError:
    #         kl_div = self.distribution().log_prob(samples) - prior.log_prob(samples)
    #         kl_div = kl_div.mean(dim=0)
    #     return kl_div
