from typing import Optional

import torch
from abismal_torch.prior.base import PriorBase
from abismal_torch.symmetry import ReciprocalASUCollection

class NormalPrior(PriorBase):
    def __init__(
        self, 
        rac: ReciprocalASUCollection, 
        loc_init: Optional[float] = 0.0, 
        scale_init: Optional[float] = 1.0, 
        **kwargs
    ):
        super().__init__(**kwargs)
        self.rac = rac
        self._loc_init = loc_init
        self._scale_init = scale_init

    def distribution(
        self, rasu_id: Optional[torch.Tensor] = None, hkl: Optional[torch.Tensor] = None
    ) -> torch.distributions.Distribution:
        """
        Args:
            rasu_id (torch.Tensor, optional): RASU IDs to get distribution for. Defaults to None,
                implicitly assumes rac.rasu_ids.
            hkl (torch.Tensor, optional): Miller indices to get distribution for. Not used in this
                prior. Only kept for compatibile API with other priors.

        Returns:
            p (torch.distributions.Distribution): Normal distribution for the specified number of
                reflections, which is rac_size if rasu_id is None and len(rasu_id) otherwise.
        """
        if rasu_id is None:
            rasu_id = self.rac.rasu_ids
        ones = torch.ones_like(rasu_id, dtype=self.rac.multiplicity.dtype)
        loc = ones * self._loc_init
        scale = ones * self._scale_init
        return torch.distributions.Normal(loc, scale)


class HalfNormalPrior(PriorBase):
    def __init__(
        self, 
        rac: ReciprocalASUCollection, 
        scale_init: Optional[float] = 1.0, 
        **kwargs
    ):
        super().__init__(**kwargs)
        self.rac = rac
        self._scale_init = scale_init

    def distribution(
        self, rasu_id: Optional[torch.Tensor] = None, hkl: Optional[torch.Tensor] = None
    ) -> torch.distributions.Distribution:
        """
        Args:
            rasu_id (torch.Tensor, optional): RASU IDs to get distribution for. Defaults to None,
                implicitly assumes rac.rasu_ids.
            hkl (torch.Tensor, optional): Miller indices to get distribution for. Not used in this
                prior. Only kept for compatibile API with other priors.

        Returns:
            p (torch.distributions.Distribution): HalfNormal distribution for the specified number of
                reflections, which is rac_size if rasu_id is None and len(rasu_id) otherwise.
        """
        if rasu_id is None:
            rasu_id = self.rac.rasu_ids
        scale = torch.ones_like(rasu_id, dtype=self.rac.multiplicity.dtype) * self._scale_init
        return torch.distributions.HalfNormal(scale)