from typing import Optional

import rs_distributions.modules as rsm
import torch

from abismal_torch.surrogate_posterior.base import PosteriorBase
from abismal_torch.symmetry import ReciprocalASUCollection


class FoldedNormalPosterior(PosteriorBase):
    def __init__(
        self,
        rac: ReciprocalASUCollection,
        loc: torch.Tensor,
        scale: torch.Tensor,
        epsilon: Optional[float] = 1e-12,
        **kwargs
    ):
        """
        Build a learnable FoldedNormal distribution from constrained initial parameters. The
            transform for the parameters is the default from the arg constraints.

        Args:
            rac (ReciprocalASUCollection): ReciprocalASUCollection object.
            loc (rsm.TransformedParameter): Learnable constrained location parameter of the distribution.
            scale (rsm.TransformedParameter): Learnable constrained scale parameter of the distribution.

        Attributes:
            rac (ReciprocalASUCollection): ReciprocalASUCollection object.
            distribution (rs_distributions.modules.DistributionModule): Learnable torch distribution object.
        """
        distribution = rsm.FoldedNormal(loc, scale)
        super().__init__(rac, distribution, epsilon, **kwargs)

    @classmethod
    def from_unconstrained_loc_and_scale(
        cls,
        rac: ReciprocalASUCollection,
        loc: torch.Tensor,
        scale: torch.Tensor,
        epsilon: Optional[float] = 1e-12,
        **kwargs
    ):
        """
        Build a learnable FoldedNormal distribution from unconstrained initial parameters. The
            transform for both loc and scale is to ensure positive values with custom epsilon.

        Args:
            rac (ReciprocalASUCollection): ReciprocalASUCollection object.
            loc (torch.Tensor): Unconstrained location parameter of the distribution.
            scale (torch.Tensor): Unconstrained scale parameter of the distribution.
            epsilon (float, optional): Epsilon value for numerical stability. Defaults to 1e-12.
        """
        transform = torch.distributions.ComposeTransform(
            [
                torch.distributions.ExpTransform(),
                torch.distributions.AffineTransform(epsilon, 1.0),
            ]
        )
        #scale_init = transform(scale)
        scale = rsm.TransformedParameter(scale, transform)
        return cls(rac, loc, scale, epsilon, **kwargs)
