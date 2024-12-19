from typing import Optional, Sequence, Tuple

import torch

from abismal_torch.distributions import DistributionBase
from abismal_torch.prior.base import PriorBase
from abismal_torch.surrogate_posterior.base import PosteriorBase
from abismal_torch.symmetry import Op


class VariationalMergingModel(torch.nn.Module):
    def __init__(
        self,
        scale_model: torch.nn.Module,
        surrogate_posterior: PosteriorBase,
        prior: PriorBase,
        likelihood: torch.nn.Module,
        mc_samples: Optional[int] = 1,
        reindexing_ops: Optional[Sequence[str]] = None,
        **kwargs,
    ) -> None:
        """
        Top-level variational merging model.

        Args:
            scale_model (torch.nn.Module): Scale model.
            surrogate_posterior (PosteriorBase): Surrogate posterior distribution model.
            prior (PriorBase): Prior distribution model.
            likelihood (torch.nn.Module): Likelihood model.
            mc_samples (int, optional): Number of Monte Carlo samples to average loss over. Defaults to 1.
            reindexing_ops (Sequence[str], optional): Reindexing operations. Defaults to identity ["x,y,z"].
            standardization_count_max (int, optional): Standardization count max. Defaults to 2_000.
        """
        super().__init__(**kwargs)
        self.likelihood = likelihood
        self.prior = prior
        self.scale_model = scale_model
        self.surrogate_posterior = surrogate_posterior
        self.mc_samples = mc_samples
        if reindexing_ops is None:
            reindexing_ops = ["x,y,z"]
        self.reindexing_ops = [Op(op) for op in reindexing_ops]

    def compute_kl_divergence(
        self,
        q: DistributionBase | torch.distributions.Distribution,
        p: DistributionBase | torch.distributions.Distribution,
        samples: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute the KL divergence between the surrogate posterior and the prior.

        Args:
            q (Distribution): The surrogate posterior.
            p (Distribution): The prior.
            samples (torch.Tensor, optional): Samples from the surrogate posterior. Only used
                if no KL divergence is implemented for the posterior-prior pair.

        Returns:
            KL divergence (torch.Tensor): A tensor of shape (Distribution's batch_shape,).
        """
        try:
            return q.kl_divergence(p, samples=samples)
        except AttributeError or NotImplementedError:
            kl_div = q.log_prob(samples) - p.log_prob(samples)
            return kl_div.mean(dim=0)

    def average_by_images(
        self, source_value: torch.Tensor, image_id: torch.Tensor
    ) -> torch.Tensor:
        """
        Average source_value along images and across mc samples.

        Args:
            source_value (torch.Tensor): A tensor of shape (n_reflns, n_samples).
            image_id (torch.Tensor): A tensor of shape (n_reflns,) that contains the
                image index for each reflection.

        Returns:
            averaged (torch.Tensor): A tensor of shape (n_images,).
        """
        n_images = image_id.max() + 1
        _, mc_samples = source_value.shape
        idx = torch.tile(image_id[:, None], (1, mc_samples))
        _averaged = torch.zeros((n_images, mc_samples))
        _averaged.scatter_add_(dim=0, index=idx, src=source_value)
        n_reflns_per_image = torch.bincount(image_id)
        averaged = _averaged.sum(dim=1) / n_reflns_per_image / mc_samples
        return averaged

    def forward(self, inputs: Tuple[torch.Tensor, ...]) -> dict:
        """
        The predicted intensities are computed as:
        ```
            Ipred = Scale_pred * |Fpred|^2
        ```
        Optionally, reindexing is performed to find the solution that minimizes
        the negative log-likelihood loss.

        Args:
            inputs (Tuple[torch.Tensor, ...]): Inputs to the model.

        Returns:
            dict: Contains the following keys:
                - ipred_avg: Average predicted intensities across MC samples
                - loss_ll: Negative log-likelihood loss
                - loss_kl: KL divergence loss
        """
        (
            image_id,
            asu_id,
            hkl_in,
            resolution,
            wavelength,
            metadata,
            iobs,
            sigiobs,
        ) = inputs

        scale = self.scale_model(
            inputs,
            image_id=image_id,
            mc_samples=self.mc_samples,
        )

        q = self.surrogate_posterior  # .distribution()
        p = self.prior.distribution()
        z = q.rsample((self.mc_samples,))  # Shape (mc_samples, rac_size)
        kl_div = self.compute_kl_divergence(q, p, samples=z)

        ll = None
        ipred = None
        hkl = None
        for op in self.reindexing_ops:
            _hkl = op(hkl_in)
            _ipred = self.surrogate_posterior.rac.gather(
                z.T, asu_id, _hkl
            )  # Shape (n_refln, mc_samples)
            _ipred = torch.square(_ipred) * scale

            _ll = self.likelihood(_ipred, iobs, sigiobs)
            _ll = self.average_by_images(_ll, image_id)  # Shape (n_images,)

            if ll is None:
                ipred = _ipred
                ll = _ll
                hkl = _hkl
            else:
                idx = _ll > ll
                ipred = torch.where(
                    idx[image_id].unsqueeze(-1), _ipred, ipred
                )  # Shape (n_refln, mc_samples)
                ll = torch.where(idx, _ll, ll)  # Shape (n_images,)
                hkl = torch.where(
                    idx[image_id].unsqueeze(-1), _hkl, hkl
                )  # Shape (n_refln, 3)

        ipred_avg = torch.mean(ipred, dim=-1)  # Shape (n_refln,)
        return {
            "ipred_avg": ipred_avg,
            "loss_nll": -ll.mean(),
            "loss_kl": kl_div.mean(),
        }
