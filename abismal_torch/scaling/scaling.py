from typing import Optional, Sequence

import rs_distributions.modules as rsm
import torch
import torch.distributions as td
import torch.nn as nn

from abismal_torch.distributions import compute_kl_divergence
from abismal_torch.layers import *


class ImageScaler(nn.Module):
    def __init__(
        self,
        mlp_width: Optional[int] = 32,
        mlp_depth: Optional[int] = 20,
        share_weights: Optional[bool] = True,
        hidden_units: Optional[int] = None,
        use_glu: Optional[bool] = False,
        activation: Optional[str | nn.Module] = "ReLU",
        scaling_posterior: Optional[
            str | type[torch.distributions.Distribution]
        ] = td.Normal,
        scaling_kl_weight: Optional[float] = 0.01,
        scaling_prior: Optional[
            str | type[torch.distributions.Distribution]
        ] = td.Laplace,
        scaling_prior_params: Optional[tuple] = (0.0, 1.0),
        epsilon: Optional[float] = 1e-12,
        **kwargs
    ) -> None:
        """
        An image scaler that uses MLP (currently only supports default FeedForward or FeedForwardGLU)
        on metadata and image embeddings to predict the parameters of the scaling posterior distribution.
        The architecture is:

        ```mermaid
        graph LR
            I(["(iobs, sigiobs)"]) --> K[concat]
            M([metadata]) --> K
            K --> B[image_linear_in]
            B --> C[mlp]
            C --> D[pool]
            D --> E["+"]
            M --> G[scale_linear_in]
            G --> E
            E --> H[scale_mlp]
            H --> I2[linear_out]
            I2 --> J(["(loc, scale)"])
        ```

        Args:
            mlp_width (int, optional): int, see MLP argument.
            mlp_depth (int, optional): int, see MLP argument.
            share_weights (bool, optional): bool, whether to share weights between the
                image and the scale MLPs.
            hidden_units (int, optional): int, see FeedForward argument.
            use_glu (bool, optional): bool, see MLP argument.
            scaling_posterior (torch.distributions.Distribution, optional): distribution class for
                the scaling posterior distribution. Defaults to a Gamma distribution.
            scaling_kl_weight (float, optional): float, the weight of the KL divergence
                loss between the scaling posterior and the scaling prior. Defaults to 0.01.
            scaling_prior (torch.distributions.Distribution, optional): instantiated distribution
                for the scaling prior. Defaults to a Laplace distribution with mean 0.0 and
                scale 1.0.
            epsilon (float, optional): float, the epsilon value for numerical stability.
                Defaults to 1e-12.
        """
        super().__init__(**kwargs)
        self.image_linear_in = CustomInitLazyLinear(mlp_width)
        self.scale_linear_in = CustomInitLazyLinear(mlp_width)
        if isinstance(scaling_posterior, str):
            self.scaling_posterior = getattr(td, scaling_posterior)
        else:
            self.scaling_posterior = scaling_posterior
        self._num_posterior_args = len(self.scaling_posterior.arg_constraints)
        self.linear_out = CustomInitLazyLinear(self._num_posterior_args)
        self.pool = ImageAverage()
        self.share_weights = share_weights
        self.mlp = MLP(
            mlp_width,
            mlp_depth,
            input_layer=False,
            hidden_units=hidden_units,
            use_glu=use_glu,
            activation=activation,
        )
        if share_weights:
            self.scale_mlp = self.mlp
        else:
            self.scale_mlp = MLP(
                mlp_width,
                mlp_depth,
                input_layer=False,
                hidden_units=hidden_units,
                use_glu=use_glu,
                activation=activation,
            )
        self.scaling_kl_weight = scaling_kl_weight
        self.epsilon = epsilon
        self.register_buffer(
            "scaling_prior_params",
            torch.tensor(scaling_prior_params, dtype=torch.float32),
        )
        self._scaling_prior = scaling_prior

    def init_scaling_prior(self, reference_data: torch.Tensor) -> None:
        if isinstance(self._scaling_prior, str):
            self.scaling_prior = getattr(td, self._scaling_prior)(
                *self.scaling_prior_params
            )
        else:
            self.scaling_prior = self._scaling_prior(*self.scaling_prior_params)

    def _create_image(self, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Create the image tensor for the MLP after properly reshaping and concatenating inputs.
        """
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
        image = torch.concat(
            (metadata, iobs, sigiobs), axis=-1
        )  # Shape (n_reflns, n_features + 2)
        return image

    def forward(
        self,
        inputs: Sequence[torch.Tensor],
        image_ids_in_this_batch: torch.Tensor,
        n_reflns_per_image: torch.Tensor,
        mc_samples: Optional[int] = 32,
    ) -> torch.Tensor:
        """
        Args:
            inputs (Sequence[torch.Tensor]): a tuple of (asu_id, ..., metadata, iobs, sigiobs)
                tensors of shape (n_reflns, n_features).
            image_ids_in_this_batch (torch.Tensor): shape (n_reflns), the image id for each
                reflection.
            n_reflns_per_image (torch.Tensor): shape (n_images), the number of reflections in each
                image.
            mc_samples (int, optional): int, number of samples to draw from the scaling posterior.
                Defaults to 32.

        Returns:
            z (torch.Tensor): shape (mc_samples, n_reflns), samples from the learned scaling
                posterior.
            kl_div (torch.Tensor): shape (n_reflns,), the weightedKL divergence between the scaling
                posterior and the scaling prior.
        """
        metadata = inputs["metadata"]
        image = self._create_image(inputs)  # Shape (n_reflns, n_features + 2)
        image_embeddings = self.image_linear_in(image)  # Shape (n_reflns, mlp_width)
        image_embeddings = self.mlp(image_embeddings)  # Shape (n_reflns, mlp_width)
        image_embeddings = self.pool(
            image_embeddings, image_ids_in_this_batch, n_reflns_per_image
        )  # Shape (n_images, mlp_width)

        scale_embeddings = self.scale_linear_in(metadata)  # Shape (n_reflns, mlp_width)
        scale_embeddings = scale_embeddings + image_embeddings[image_ids_in_this_batch]
        scale_embeddings = self.scale_mlp(
            scale_embeddings
        )  # Shape (n_reflns, mlp_width)
        scaling_params = self.linear_out(scale_embeddings)  # Shape (n_reflns, _num_posterior_args)

        # softplus transform
        # loc, scale = scaling_params.unbind(dim=-1)
        # scale = torch.nn.functional.softplus(scale) + self.epsilon
        # print(f"scaling_params shape: {scaling_params.shape}", flush=True)
        scale = scaling_params[..., -1]
        scale = torch.nn.functional.softplus(scale) + self.epsilon
        if self._num_posterior_args == 1:
            q = self.scaling_posterior(scale.squeeze(-1))
        else:
            loc, _ = scaling_params.unbind(dim=-1)
            q = self.scaling_posterior(loc, scale)
        z = q.rsample(sample_shape=(mc_samples,))  # Shape (mc_samples, n_reflns)
        # print(f"z shape: {z.shape}", flush=True)
        if not hasattr(self, "scaling_prior"):
            self.init_scaling_prior(scaling_params)
        p = self.scaling_prior.expand((len(scaling_params),))
        kl_div = compute_kl_divergence(q, p, samples=z) * self.scaling_kl_weight
        return {
            "z": torch.t(z),  # Shape (n_reflns, mc_samples)
            "kl_div": kl_div,
        }  # Shape (n_reflns,)
