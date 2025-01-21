from typing import Iterator, Optional

import numpy as np
import reciprocalspaceship as rs
import torch
from rs_distributions.modules import DistributionModule

from abismal_torch.distributions import PosteriorDistributionBase
from abismal_torch.symmetry import ReciprocalASUCollection


class PosteriorBase(torch.nn.Module):
    def __init__(
        self,
        rac: ReciprocalASUCollection,
        distribution: DistributionModule | PosteriorDistributionBase,
        **kwargs,
    ):
        """
        Base class for surrogate posteriors.

        Args:
            rac (ReciprocalASUCollection): ReciprocalASUCollection.
            distribution (rs.DistributionModule | PosteriorDistributionBase): Learnable distribution object.
            observed (torch.Tensor): a boolean tensor of shape (rac_size,) indicating which unique reflections
            are observed.
        """
        super().__init__(**kwargs)
        self.rac = rac
        self.distribution = distribution
        self.register_buffer(
            "observed", torch.zeros(self.rac.rac_size, dtype=torch.bool)
        )

    def rsample(self, *args, **kwargs) -> torch.Tensor:
        return self.distribution.rsample(*args, **kwargs)

    def log_prob(self, z: torch.Tensor) -> torch.Tensor:
        return self.distribution.log_prob(z)

    def update_observed(self, rasu_id: torch.Tensor, H: torch.Tensor) -> None:
        """
        Update the observed buffer with the indices of the reflections in the training batch.
        This is previously named as register_seen() in tensorflow abismal.

        Args:
            rasu_id (torch.Tensor): A tensor of shape (n_refln,) that contains the
                rasu ID of each reflection.
            H (torch.Tensor): A tensor of shape (n_refln, 3).
        """
        h, k, l = H.T
        observed_idx = self.rac.reflection_id_grid[rasu_id, h, k, l]
        self.observed[observed_idx] = True

    def to_dataset(self, only_observed: Optional[bool] = True) -> Iterator[rs.DataSet]:
        mean = self.distribution.mean.detach().numpy()
        std = self.distribution.stddev.detach().numpy()

        h, k, l = self.rac.H_rasu.numpy().T
        data = {
            "H": rs.DataSeries(h, dtype="H"),
            "K": rs.DataSeries(k, dtype="H"),
            "L": rs.DataSeries(l, dtype="H"),
            "F": rs.DataSeries(mean, dtype="F"),
            "SIGF": rs.DataSeries(std, dtype="Q"),
        }

        # Only save reflections in the ASU for each dataset
        for i, rasu in enumerate(self.rac):
            ds = rs.DataSet(
                data,
                merged=True,
                cell=rasu.cell,
                spacegroup=rasu.spacegroup,
            )
            idx = self.rac.rasu_ids.numpy() == i
            print("For rasu_id", i, "out of size", self.rac.rac_size)
            print(f"Number of reflections before only_observed: {np.sum(idx)}")
            if only_observed:
                idx = idx & self.observed.numpy()
            print(f"Number of reflections after only_observed: {np.sum(idx)}")
            out = ds[idx]
            out = out.set_index(["H", "K", "L"])
            if rasu.anomalous:
                out = out.unstack_anomalous()
                keys = [
                    "F(+)",
                    "SIGF(+)",
                    "F(-)",
                    "SIGF(-)",
                ]
                out = out[keys]
            yield out
