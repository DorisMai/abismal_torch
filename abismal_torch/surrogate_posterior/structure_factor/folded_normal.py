from typing import Optional

import rs_distributions.modules as rsm
import rs_distributions.distributions as rsd
import torch

from abismal_torch.surrogate_posterior.base import PosteriorBase
from abismal_torch.symmetry import ReciprocalASUCollection


class FoldedNormalPosterior(PosteriorBase):
    def __init__(
        self,
        rac: ReciprocalASUCollection,
        loc: torch.Tensor | rsm.TransformedParameter,
        scale: torch.Tensor | rsm.TransformedParameter,
        epsilon: Optional[float] = 1e-12,
        transform: Optional[ str | type[torch.distributions.transforms.Transform]] = 'ExpTransform',
        **kwargs
    ):
        """
        Build a learnable FoldedNormal distribution from constrained initial parameters. The
            transform for the parameters is the default from the arg constraints.

        Args:
            rac (ReciprocalASUCollection): ReciprocalASUCollection object.
            loc (torch.Tensor | rsm.TransformedParameter): Learnable constrained location parameter.
            scale (torch.Tensor | rsm.TransformedParameter): Learnable constrained scale parameter.
            epsilon (float, optional): Epsilon value for numerical stability, only used if user specifies
                a non-default transform. Defaults to 1e-12.
            transform (str | type[torch.distributions.transforms.Transform], optional): Custom transform 
                associated with the scale parameter (if not already an rsm.TransformedParameter). Defaults
                to ExpTransform and then chained with an AffineTransform to ensure positive values.

        Attributes:
            rac (ReciprocalASUCollection): ReciprocalASUCollection object.
            distribution (rs_distributions.modules.DistributionModule): Learnable torch distribution object.
        """
        if transform is not None and not isinstance(scale, rsm.TransformedParameter):
            if isinstance(transform, str):
                transform = getattr(torch.distributions.transforms, transform)
                transform = torch.distributions.ComposeTransform(
                    [
                        transform(),
                        torch.distributions.AffineTransform(epsilon, 1.0),
                    ]
                )
            scale = rsm.TransformedParameter(scale, transform)            
        distribution = rsm.FoldedNormal(loc, scale)
        super().__init__(rac, distribution, epsilon, **kwargs)

    def lazy_distribution(self, rasu_id: Optional[torch.Tensor] = None, hkl: Optional[torch.Tensor] = None):
        if rasu_id is None:
            rasu_id = self.rac.rasu_ids
        if hkl is None:
            hkl = self.rac.H_rasu
        loc_transform = self.distribution._transformed_loc.transform
        scale_transform = self.distribution._transformed_scale.transform
        lazy_loc = self.rac.gather(self.distribution._transformed_loc._value, rasu_id, hkl)
        lazy_scale = self.rac.gather(self.distribution._transformed_scale._value, rasu_id, hkl)
        return rsd.FoldedNormal(loc_transform(lazy_loc), scale_transform(lazy_scale))

    @classmethod
    def from_unconstrained_loc_and_scale(
        cls,
        rac: ReciprocalASUCollection,
        loc: torch.Tensor,
        scale_unconstrained: torch.Tensor,
        epsilon: Optional[float] = 1e-12,
        transform: Optional[ str | type[torch.distributions.transforms.Transform]] = 'ExpTransform',
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
            transform (str | type[torch.distributions.transforms.Transform], optional): Transform to apply
                to the scale parameter. If not provided, the default transform is based on arg constraints.
                Defaults to ExpTransform and then chained with an AffineTransform to ensure positive values.
        """
        if transform is None:
            constraint = rsm.FoldedNormal.arg_constraints['scale']
            transform = torch.distributions.constraint_registry.transform_to(constraint)
        else:
            if isinstance(transform, str):
                transform = getattr(torch.distributions.transforms, transform)
                transform = torch.distributions.ComposeTransform(
                    [
                        transform(),
                        torch.distributions.AffineTransform(epsilon, 1.0),
                    ]
                )
        scale = rsm.TransformedParameter(transform(scale_unconstrained), transform)
        return cls(rac, loc, scale, epsilon, transform=None, **kwargs)