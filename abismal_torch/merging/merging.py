from typing import Optional, Sequence, Tuple

import torch

from abismal_torch.distributions import compute_kl_divergence
from abismal_torch.layers import LazyWelfordStandardization as Standardization
from abismal_torch.layers.average import ImageAverage
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

        Attributes:
            standardize_intensity (torch.nn.Module): Standardization layer for intensity data.
            standardize_metadata (torch.nn.Module): Standardization layer for metadata data.
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
        self.standardize_intensity = Standardization(center=False)
        self.standardize_metadata = Standardization(center=True)
        self.pool = ImageAverage()

    def standardize_inputs(
        self, inputs: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        metadata = inputs["metadata"]
        iobs = (
            inputs["iobs"].unsqueeze(-1)
            if inputs["iobs"].dim() == 1
            else inputs["iobs"]
        )
        sigiobs = (
            inputs["sigiobs"].unsqueeze(-1)
            if inputs["sigiobs"].dim() == 1
            else inputs["sigiobs"]
        )
        metadata = self.standardize_metadata(metadata)
        iobs = self.standardize_intensity(iobs)
        sigiobs = self.standardize_intensity.standardize(
            sigiobs
        )  # error propagation without updating running stats
        inputs["metadata"] = metadata
        inputs["iobs"] = iobs
        inputs["sigiobs"] = sigiobs
        return inputs

    def forward(self, inputs: dict[str, torch.Tensor]) -> dict:
        """
        The predicted intensities are computed as:
        ```
            Ipred = Scale_pred * |Fpred|^2
        ```
        Optionally, reindexing is performed to find the solution that minimizes
        the negative log-likelihood loss.

        Args:
            inputs (dict[str, torch.Tensor]): Inputs to the model.

        Returns:
            dict: Contains the following keys:
                - ipred_avg: Average predicted intensities across MC samples
                - loss_ll: Negative log-likelihood loss
                - loss_kl: KL divergence loss
        """
        inputs = self.standardize_inputs(inputs)
        image_id = inputs["image_id"]
        rasu_id = inputs["rasu_id"]
        hkl_in = inputs["hkl_in"]
        iobs = inputs["iobs"]
        sigiobs = inputs["sigiobs"]

        # Scaling model
        scale_outputs = self.scale_model(
            inputs,
            image_id=image_id,
            mc_samples=self.mc_samples,
        )
        scale = scale_outputs["z"]
        scale_kl_div = scale_outputs["kl_div"]

        # Structure factor
        q = self.surrogate_posterior
        p = self.prior.distribution()
        z = q.rsample((self.mc_samples,))  # Shape (mc_samples, rac_size)
        kl_div = compute_kl_divergence(q, p, samples=z)
        # for debug
        if torch.any(z == 0):
            from IPython import embed
            embed(colors='linux')

        # Reindexing for optimal likelihood
        ll = None
        ipred = None
        hkl = None
        reflns_per_image = None
        for op in self.reindexing_ops:
            _hkl = op(hkl_in)
            _ipred = self.surrogate_posterior.rac.gather(
                z.T, rasu_id, _hkl
            )  # Shape (n_refln, mc_samples)
            _ipred = torch.square(_ipred) * scale

            _ll = self.likelihood(_ipred, iobs, sigiobs)
            _ll, _, reflns_per_image = self.pool(_ll, image_id)  # Shape (n_images, mc_samples), _, (n_images,)
            _ll = _ll.mean(dim=1) # Shape (n_images,)

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

        # Reweight likelihood by number of reflections in each image
        ll = ll * reflns_per_image / reflns_per_image.sum()
        
        # if ipred or ll any is not finite, drop into IPython
        if not torch.all(torch.isfinite(ipred)) or not torch.all(torch.isfinite(ll)):
            from IPython import embed
            embed(colors='linux')

        return {
            "loss_nll": -ll, #shape (n_images,)
            "loss_kl": kl_div, #shape (rac_size,)
            "scale_kl_div": scale_kl_div, #shape (n_reflns,)
            "ipred_avg": torch.mean(ipred, dim=-1), #shape (n_reflns,)
            "hkl": hkl, #shape (n_reflns, 3)
            "z": z, #shape (mc_samples, rac_size)
            "scale": scale, #shape (mc_samples, n_reflns)
        }