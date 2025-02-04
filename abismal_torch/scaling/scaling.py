from typing import Optional, Sequence

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
        scaling_posterior: Optional[torch.distributions.Distribution] = td.Gamma,
        scaling_kl_weight: Optional[float] = 0.01,
        scaling_prior: Optional[torch.distributions.Distribution] = td.Laplace(
            0.0, 1.0
        ),
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
        """
        super().__init__(**kwargs)
        self.image_linear_in = CustomInitLazyLinear(mlp_width)
        self.scale_linear_in = CustomInitLazyLinear(mlp_width)
        num_distribution_args = len(scaling_posterior.arg_constraints)
        self.linear_out = CustomInitLazyLinear(num_distribution_args)
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
        self.scaling_posterior = scaling_posterior
        self.scaling_kl_weight = scaling_kl_weight
        self.scaling_prior = scaling_prior

    def _create_image(self, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Create the image tensor for the MLP after properly reshaping and concatenating inputs.
        """
        metadata = inputs["metadata"]
        iobs = inputs["iobs"]
        sigiobs = inputs["sigiobs"]
        if len(iobs.shape) == 1:
            iobs = iobs[:, None]
        if len(sigiobs.shape) == 1:
            sigiobs = sigiobs[:, None]
        image = torch.concat(
            (metadata, iobs, sigiobs), axis=-1
        )  # Shape (n_reflns, n_features + 2)
        return image

    def forward(
        self,
        inputs: Sequence[torch.Tensor],
        image_id: torch.Tensor,
        mc_samples: Optional[int] = 32,
    ) -> torch.Tensor:
        """
        Args:
            inputs (Sequence[torch.Tensor]): a tuple of (asu_id, ..., metadata, iobs, sigiobs)
                tensors of shape (n_reflns, n_features).
            image_id (torch.Tensor): shape (n_reflns), the image id for each reflection.
            mc_samples (int, optional): int, number of samples to draw from the scaling posterior.
                Defaults to 32.

        Returns:
            z (torch.Tensor): shape (mc_samples, n_reflns), samples from the learned scaling
                posterior.
        """
        metadata = inputs["metadata"]
        image = self._create_image(inputs)  # Shape (n_reflns, n_features + 2)
        image_embeddings = self.image_linear_in(image)  # Shape (n_reflns, mlp_width)
        image_embeddings = self.mlp(image_embeddings)  # Shape (n_reflns, mlp_width)
        image_embeddings = self.pool(
            image_embeddings, image_id
        )  # Shape (n_images, mlp_width)

        scale_embeddings = self.scale_linear_in(metadata)  # Shape (n_reflns, mlp_width)
        scale_embeddings = (
            scale_embeddings + image_embeddings[image_id]
        )  # Shape (n_reflns, mlp_width)
        scale_embeddings = self.scale_mlp(
            scale_embeddings
        )  # Shape (n_reflns, mlp_width)
        scaling_params = self.linear_out(scale_embeddings)  # Shape (n_reflns, 2)

        # transform scaling_params to satisfy distribution constraints
        for i, constraint in enumerate(self.scaling_posterior.arg_constraints.values()):
            scaling_params[:, i] = torch.distributions.transform_to(constraint)(
                scaling_params[:, i]
            )

        q = self.scaling_posterior(*scaling_params.unbind(dim=-1))
        z = q.rsample(sample_shape=(mc_samples,))  # Shape (mc_samples, n_reflns)
        p = self.scaling_prior.expand((len(scaling_params),))
        kl_div = compute_kl_divergence(q, p, samples=z)
        return {"z": torch.t(z), "kl_div": kl_div}  # Shape (n_reflns, mc_samples)
