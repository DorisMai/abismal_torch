from typing import Optional

import torch

from abismal_torch.distributions import DistributionBase


class PriorBase(torch.nn.Module):
    def distribution(
        self, rasu_id: Optional[torch.Tensor] = None, hkl: Optional[torch.Tensor] = None
    ) -> DistributionBase:
        raise NotImplementedError(
            "Derived classes must implement distribution(asu_id, hkl) -> DistributionBase"
        )
