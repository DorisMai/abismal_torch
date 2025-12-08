from typing import Optional

import rs_distributions.modules as rsm
import torch
import torch.distributions as td
from abismal_torch.surrogate_posterior.base import PosteriorBase
from abismal_torch.symmetry import ReciprocalASUCollection


class MultivariateNormalPosterior(PosteriorBase):
    def __init__(
        self,
        rac: ReciprocalASUCollection,
        loc: torch.Tensor | rsm.TransformedParameter,
        cov_factor: torch.Tensor | rsm.TransformedParameter,
        cov_diag: torch.Tensor | rsm.TransformedParameter,
        epsilon: Optional[float] = 1e-12,
        transform: Optional[ str | type[torch.distributions.transforms.Transform]] = 'ExpTransform',
        **kwargs
    ):
        """
        Build a learnable Low-rank Multivariate Normal distribution from constrained initial parameters. The
            transform for the parameters is the default from the arg constraints.

        Args:
            rac (ReciprocalASUCollection): ReciprocalASUCollection object.
            loc (torch.Tensor | rsm.TransformedParameter): Learnable constrained location parameter.
            cov_factor (torch.Tensor | rsm.TransformedParameter): Learnable constrained covariance factor parameter.
            cov_diag (torch.Tensor | rsm.TransformedParameter): Learnable constrained covariance diagonal parameter.
            epsilon (float, optional): Epsilon value for numerical stability, only used if user specifies a
                non-default transform. Defaults to 1e-12.
            transform (str | type[torch.distributions.transforms.Transform], optional): Transform for the covariance
                diagonal parameter (if not already an rsm.TransformedParameter). Defaults to ExpTransform
                and then chained with an AffineTransform to ensure positive values.

        Attributes:
            rac (ReciprocalASUCollection): ReciprocalASUCollection object.
            distribution (rs_distributions.modules.DistributionModule): Learnable torch distribution object.
        """
        if transform is not None and not isinstance(cov_diag, rsm.TransformedParameter):
            if isinstance(transform, str):
                transform = getattr(torch.distributions.transforms, transform)
                transform = torch.distributions.ComposeTransform(
                    [
                        transform(),
                        torch.distributions.AffineTransform(epsilon, 1.0),
                    ]
                )
            cov_diag = rsm.TransformedParameter(cov_diag, transform)            
        distribution = rsm.LowRankMultivariateNormal(loc, cov_factor, cov_diag)
        super().__init__(rac, distribution, epsilon, **kwargs)

    def lazy_distribution(self, rasu_id: Optional[torch.Tensor] = None, hkl: Optional[torch.Tensor] = None):
        if rasu_id is None:
            rasu_id = self.rac.rasu_ids
        if hkl is None:
            hkl = self.rac.H_rasu
        loc_transform = self.distribution._transformed_loc.transform
        cov_factor_transform = self.distribution._transformed_cov_factor.transform
        cov_diag_transform = self.distribution._transformed_cov_diag.transform
        lazy_loc = self.rac.gather(self.distribution._transformed_loc._value, rasu_id, hkl)
        lazy_cov_factor = self.rac.gather(self.distribution._transformed_cov_factor._value, rasu_id, hkl)
        lazy_cov_diag = self.rac.gather(self.distribution._transformed_cov_diag._value, rasu_id, hkl)
        return td.LowRankMultivariateNormal(loc_transform(lazy_loc), cov_factor_transform(lazy_cov_factor), cov_diag_transform(lazy_cov_diag))

    @classmethod
    def from_unconstrained_loc_and_scale(
        cls,
        rac: ReciprocalASUCollection,
        loc: torch.Tensor,
        cov_facor: torch.Tensor,
        cov_diag_unconstrained: torch.Tensor,
        epsilon: Optional[float] = 1e-12,
        transform: Optional[ str | type[torch.distributions.transforms.Transform]] = 'ExpTransform',
        **kwargs
    ):
        """
        Build a learnable Low-rank Multivariate Normal distribution from unconstrained initial parameters.
        
        Args:
            rac (ReciprocalASUCollection): ReciprocalASUCollection object.
            loc (torch.Tensor): Unconstrained location parameter of the distribution.
            cov_factor (torch.Tensor): Unconstrained covariance factor parameter of the distribution.
            cov_diag (torch.Tensor): Unconstrained covariance diagonal parameter of the distribution.
            epsilon (float, optional): Epsilon value for numerical stability. Defaults to 1e-12.
            transform (str | type[torch.distributions.transforms.Transform], optional): Transform to 
                apply to the covariance diagonal parameter. If not, the default transform is based on
                arg constraints. Defaults to ExpTransform and then chained with an AffineTransform to
                ensure positive values.
        """
        if transform is None:
            constraint = rsm.LowRankMultivariateNormal.arg_constraints['cov_diag']
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
        cov_diag = rsm.TransformedParameter(transform(cov_diag_unconstrained), transform)
        return cls(rac, loc, cov_facor, cov_diag, epsilon=epsilon, transform=None, **kwargs)