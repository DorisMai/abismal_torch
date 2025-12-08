from typing import Optional

import rs_distributions.modules as rsm
import rs_distributions.distributions as rsd
import torch
import torch.distributions as td

from abismal_torch.surrogate_posterior.base import PosteriorBase
from abismal_torch.symmetry import ReciprocalASUCollection


class LocScalePosterior(PosteriorBase):

    __SUPPORTED_DISTRIBUTIONS__ = {
        "FoldedNormal": (["loc", "scale"], rsd.FoldedNormal),
        "Normal": (["loc", "scale"], td.Normal),
        "Rice": (["nu", "sigma"], rsd.Rice),
        "Gamma": (["concentration", "rate"], td.Gamma),
    }

    def __init__(
        self,
        rac: ReciprocalASUCollection,
        rsm_distribution: str,
        param1: torch.Tensor | rsm.TransformedParameter,
        param2: torch.Tensor | rsm.TransformedParameter,
        unconstrained: bool = False,
        epsilon: Optional[float] = 1e-12,
        transform: Optional[ str | type[td.transforms.Transform]] = 'ExpTransform',
        **kwargs
    ):
        """
        Build a learnable distribution with two parameters from constrained initial values. 

        Args:
            rac (ReciprocalASUCollection): ReciprocalASUCollection object.
            rsm_distribution (str): rs_distributions module to use. Only supports the following distributions: 
                "FoldedNormal", "Normal", "Rice", "Gamma".
            param1 (torch.Tensor | rsm.TransformedParameter): Learnable constrained first parameter of the distribution.
            param2 (torch.Tensor | rsm.TransformedParameter): Learnable constrained second parameter of the distribution.
            epsilon (float, optional): Epsilon value for numerical stability, only used if user specifies a
                non-default transform. Defaults to 1e-12.
            transform (str | td.transforms.Transform, optional): Positive transform applied to 
                the parameters (if needed). Only used if not None and the parameters are not already instances of
                rsm.TransformedParameter. Defaults to ExpTransform and then chained with an AffineTransform to
                ensure positive values.

        Attributes:
            rac (ReciprocalASUCollection): ReciprocalASUCollection object.
            distribution (rs_distributions.modules.DistributionModule): Learnable torch distribution object.
        """
        if rsm_distribution not in self.__SUPPORTED_DISTRIBUTIONS__.keys():
            raise ValueError(f"Unsupported distribution: {rsm_distribution}")
        param_names, _ = self.__SUPPORTED_DISTRIBUTIONS__[rsm_distribution]
        name_to_param = dict(zip(param_names, [param1, param2]))
        rsm_distribution_class = getattr(rsm.distribution, rsm_distribution)
        if unconstrained or (transform is not None):
            for name, param in name_to_param.items():
                constraint = rsm_distribution_class.arg_constraints[name]
                transform = self._get_transform(constraint, transform)
                if self._needs_positive_transform(constraint):
                    name_to_param[name] = self._make_transformed_parameter(
                        param, 
                        transform, 
                        epsilon=epsilon, 
                        unconstrained=unconstrained, 
                    )
                else:
                    name_to_param[name] = self._make_transformed_parameter(
                        param, 
                        transform, 
                        unconstrained=unconstrained, 
                    )
        distribution = rsm_distribution_class(**name_to_param)
        super().__init__(rac, distribution, epsilon, **kwargs)
        self.distribution_name = rsm_distribution

    def lazy_distribution(self, rasu_id: Optional[torch.Tensor] = None, hkl: Optional[torch.Tensor] = None):
        if rasu_id is None:
            rasu_id = self.rac.rasu_ids
        if hkl is None:
            hkl = self.rac.H_rasu
        loc_transform = self.distribution._transformed_loc.transform
        scale_transform = self.distribution._transformed_scale.transform
        lazy_loc = self.rac.gather(self.distribution._transformed_loc._value, rasu_id, hkl)
        lazy_scale = self.rac.gather(self.distribution._transformed_scale._value, rasu_id, hkl)
        
        param_names, lazy_distribution_class = self.__SUPPORTED_DISTRIBUTIONS__[self.distribution_name]
        lazy_name_to_param = dict(zip(param_names, [loc_transform(lazy_loc), scale_transform(lazy_scale)]))
        return lazy_distribution_class(**lazy_name_to_param)
        

    @classmethod
    def from_unconstrained_params(
        cls,
        rac: ReciprocalASUCollection,
        rsm_distribution: str,
        param1_unconstrained: Optional[torch.Tensor] = None,
        param2_unconstrained: Optional[torch.Tensor] = None,
        epsilon: Optional[float] = 1e-12,
        transform: Optional[ str | td.transforms.Transform] = 'ExpTransform',
        **kwargs
    ):
        """
        Build a learnable distribution from unconstrained initial parameters.
        """
        if param1_unconstrained is None:
            param1_unconstrained = torch.ones_like(rac.multiplicity)
        if param2_unconstrained is None:
            param2_unconstrained = param1_unconstrained * 0.1
        return cls(rac, rsm_distribution, param1_unconstrained, param2_unconstrained, unconstrained=True, epsilon=epsilon, transform=transform, **kwargs)


    @staticmethod
    def _needs_positive_transform(
        constraint: td.constraints.Constraint,
    ) -> bool:
        return (
            isinstance(constraint, td.constraints._GreaterThan)
            or isinstance(constraint, td.constraints._GreaterThanEq)
        ) and constraint.lower_bound == 0

    @staticmethod
    def _get_transform(
        constraint: Optional[td.constraints.Constraint] = None,
        transform: Optional[str | type[td.transforms.Transform]] = None,
    ) -> td.transforms.Transform:
        if transform is None:
            transform = td.constraint_registry.transform_to(constraint)
        elif isinstance(transform, str):
            transform = getattr(td.transforms, transform)
        else:
            transform = transform
        return transform

    @staticmethod
    def _make_transformed_parameter(
        param: torch.Tensor,
        transform: td.transforms.Transform,
        unconstrained: bool = False,
        epsilon: Optional[float] = 1e-12,
    ) -> rsm.TransformedParameter:
        if isinstance(param, rsm.TransformedParameter):
            return param
        if epsilon is not None:
            transform = td.ComposeTransform(
                [
                    transform(),
                    td.AffineTransform(epsilon, 1.0),
                ]
            )
        if unconstrained:
            return rsm.TransformedParameter(transform(param), transform)
        else:
            return rsm.TransformedParameter(param, transform)