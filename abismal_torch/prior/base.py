from typing import Optional

import torch


class DistributionBase:
    def log_prob(self, z: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Derived classes must implement log_prob()")


class PriorBase(torch.nn.Module):
    def distribution(
        self, asu_id: Optional[torch.Tensor] = None, hkl: Optional[torch.Tensor] = None
    ) -> DistributionBase:
        raise NotImplementedError(
            "Derived classes must implement distribution(asu_id, hkl) -> DistributionBase"
        )
