import torch
from abismal_torch.distributions import PosteriorDistributionBase
import torch.distributions.constraints as constraints

class DeltaDistribution(PosteriorDistributionBase):
    arg_constraints = {"loc": constraints.real}
    support = constraints.real
    has_rsample = True

    def __init__(self, loc: torch.Tensor):
        super().__init__()
        self.loc = loc

    def rsample(self, sample_shape: tuple[int, ...] = ()) -> torch.Tensor:
        new_size = sample_shape + self.loc.shape
        return self.loc.expand(new_size).squeeze(-1)