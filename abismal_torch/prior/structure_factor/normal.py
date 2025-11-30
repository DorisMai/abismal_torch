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
        ones = torch.ones_like(self.rac.centric)
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
        scale = torch.ones_like(self.rac.centric) * self._scale_init
        return torch.distributions.HalfNormal(scale)