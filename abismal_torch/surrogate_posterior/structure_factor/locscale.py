from typing import Optional

import rs_distributions.modules as rsm
import torch

from abismal_torch.surrogate_posterior.base import PosteriorBase
from abismal_torch.symmetry import ReciprocalASUCollection


class LocScalePosterior(PosteriorBase):

    SUPPORTED_DISTRIBUTIONS = {
        rsm.FoldedNormal: ["loc", "scale"],
        rsm.Normal: ["loc", "scale"],
        rsm.Rice: ["nu", "sigma"],
        rsm.Gamma: ["concentration", "rate"],
    }

    def __init__(
        self,
        rac: ReciprocalASUCollection,
        rsm_distribution: str | rsm.DistributionModule,
        param1: torch.Tensor | rsm.TransformedParameter,
        param2: torch.Tensor | rsm.TransformedParameter,
        epsilon: Optional[float] = 1e-12,
        transform: Optional[ str | type[torch.distributions.transforms.Transform]] = 'ExpTransform',
        **kwargs
    ):
        """
        Build a learnable distribution with two parameters from constrained initial values. 

        Args:
            rac (ReciprocalASUCollection): ReciprocalASUCollection object.
            rsm_distribution (str | rsm.DistributionModule): rs_distributions module to use. Only supports the
                following distributions: FoldedNormal, Normal, Rice, Gamma.
            param1 (torch.Tensor | rsm.TransformedParameter): Learnable constrained first parameter of the distribution.
            param2 (torch.Tensor | rsm.TransformedParameter): Learnable constrained second parameter of the distribution.
            epsilon (float, optional): Epsilon value for numerical stability, only used if user specifies a
                non-default transform. Defaults to 1e-12.
            transform (str | type[torch.distributions.transforms.Transform], optional): Positive transform applied to 
                the parameters (if needed). Only used if not None and the parameters are not already instances of
                rsm.TransformedParameter. Defaults to ExpTransform and then chained with an AffineTransform to
                ensure positive values.

        Attributes:
            rac (ReciprocalASUCollection): ReciprocalASUCollection object.
            distribution (rs_distributions.modules.DistributionModule): Learnable torch distribution object.
        """
        if isinstance(rsm_distribution, str):
            rsm_distribution = getattr(rsm.distribution, rsm_distribution)
        if rsm_distribution not in self.SUPPORTED_DISTRIBUTIONS.keys():
            raise ValueError(f"Unsupported distribution: {rsm_distribution}")
        name_to_param = dict(zip(self.SUPPORTED_DISTRIBUTIONS[rsm_distribution], [param1, param2]))
        if transform is not None:
            for name, param in name_to_param.items():
                constraint = rsm_distribution.arg_constraints[name]
                if self._needs_positive_transform(constraint):
                    transform = self._get_transform(constraint, transform, epsilon)
                    name_to_param[name] = self._make_transformed_parameter(param, transform, unconstrained=False)
        distribution = rsm_distribution(**name_to_param)
        super().__init__(rac, distribution, epsilon, **kwargs)

    @classmethod
    def from_unconstrained_params(
        cls,
        rac: ReciprocalASUCollection,
        rsm_distribution: str | rsm.DistributionModule,
        param1_unconstrained: Optional[torch.Tensor] = None,
        param2_unconstrained: Optional[torch.Tensor] = None,
        epsilon: Optional[float] = 1e-12,
        transform: Optional[ str | type[torch.distributions.transforms.Transform]] = 'ExpTransform',
        **kwargs
    ):
        """
        Build a learnable distribution from unconstrained initial parameters.
        """
        if isinstance(rsm_distribution, str):
            rsm_distribution = getattr(rsm.distribution, rsm_distribution)
        if param1_unconstrained is None:
            param1_unconstrained = torch.ones_like(rac.centric)
        if param2_unconstrained is None:
            param2_unconstrained = param1_unconstrained * 0.1
        name_to_param = dict(zip(cls.SUPPORTED_DISTRIBUTIONS[rsm_distribution], [param1_unconstrained, param2_unconstrained]))
        for name, param in name_to_param.items():
            constraint = rsm_distribution.arg_constraints[name]
            if cls._needs_positive_transform(constraint):
                transform = cls._get_transform(constraint, transform, epsilon)
                name_to_param[name] = cls._make_transformed_parameter(param, transform, unconstrained=True)
        return cls(rac, rsm_distribution, param1_unconstrained, param2_unconstrained, epsilon=epsilon, transform=None, **kwargs)


    @staticmethod
    def _needs_positive_transform(
        constraint: torch.distributions.constraints.Constraint,
    ) -> bool:
        return (
            isinstance(constraint, torch.distributions.constraints._GreaterThan)
            or isinstance(constraint, torch.distributions.constraints._GreaterThanEq)
        ) and constraint.lower_bound == 0

    @staticmethod
    def _get_transform(
        constraint: torch.distributions.constraints.Constraint,
        transform: Optional[str] = None,
        epsilon: Optional[float] = 1e-12,
    ) -> torch.distributions.transforms.Transform:
        if transform is None:
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
        return transform

    @staticmethod
    def _make_transformed_parameter(
        param: torch.Tensor,
        transform: torch.distributions.transforms.Transform,
        unconstrained: bool = False,
    ) -> rsm.TransformedParameter:
        if isinstance(param, rsm.TransformedParameter):
            return param
        if unconstrained:
            return rsm.TransformedParameter(transform(param), transform)
        else:
            return rsm.TransformedParameter(param, transform)