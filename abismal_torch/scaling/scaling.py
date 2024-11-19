from typing import Optional, Sequence

import torch
import torch.nn as nn
from rs_distributions import distributions as rsd
from torch.distributions import Exponential

from abismal_torch.layers import *


class ImageScaler(nn.Module):
    def __init__(
        self,
        mlp_width: Optional[int] = 32,
        mlp_depth: Optional[int] = 20,
        share_weights: Optional[bool] = True,
        hidden_units: Optional[int] = None,
        use_glu: Optional[bool] = False,
        scaling_posterior: Optional[
            torch.distributions.Distribution
        ] = rsd.FoldedNormal,
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
            scaling_posterior (torch.distributions.Distribution, optional): the scaling
                posterior distribution.
        """
        super().__init__(**kwargs)
        self.image_linear_in = CustomInitLazyLinear(mlp_width)
        self.scale_linear_in = CustomInitLazyLinear(mlp_width)
        self.linear_out = CustomInitLazyLinear(2)
        self.pool = ImageAverage()
        self.share_weights = share_weights
        self.mlp = MLP(
            mlp_width,
            mlp_depth,
            input_layer=False,
            hidden_units=hidden_units,
            use_glu=use_glu,
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
            )
        self.scaling_posterior = scaling_posterior

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
            mc_samples (int): int, number of samples to draw from the scaling posterior.

        Returns:
            z (torch.Tensor): shape (mc_samples, n_reflns), samples from the learned scaling
                posterior.
        """
        metadata, iobs, sigiobs = inputs[-3], inputs[-2], inputs[-1]
        image = torch.concat(
            (metadata, iobs, sigiobs), axis=-1
        )  # Shape (n_reflns, n_features + 2)

        image_embeddings = self.image_linear_in(image)  # Shape (n_reflns, mlp_width)
        image_embeddings = self.mlp(image_embeddings)  # Shape (n_reflns, mlp_width)
        print(image_embeddings.shape)
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

        location, scale = scaling_params[:, 0], scaling_params[:, 1]
        scale_transform = torch.distributions.transform_to(
            rsd.FoldedNormal.arg_constraints["scale"]
        )
        q = self.scaling_posterior(loc=location, scale=scale_transform(scale))
        z = q.sample(sample_shape=(mc_samples,))  # Shape (mc_samples, n_reflns)
        z = torch.t(z)  # Shape (n_reflns, mc_samples)
        return z
