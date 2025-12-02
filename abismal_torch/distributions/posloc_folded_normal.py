from typing import Optional
from rs_distributions.distributions import FoldedNormal
import torch
import torch.distributions as td

class PosLocFoldedNormal(FoldedNormal):
    arg_constraints = {"loc": td.constraints._GreaterThan(lower_bound=0.0),
                       "scale": td.constraints._GreaterThan(lower_bound=0.0)}
    support = td.constraints.positive

    def __init__(
        self, 
        loc: torch.Tensor, 
        scale: torch.Tensor, 
        validate_args: Optional[bool] = None, 
        epsilon: Optional[float] = 1e-12):
        super().__init__(loc, scale, validate_args)
        self.epsilon = epsilon

    def sample(self, sample_shape: tuple[int, ...] = torch.Size()) -> torch.Tensor:
        return super().sample(sample_shape) + self.epsilon