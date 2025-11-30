from typing import Any, Optional, Sequence

from rs_distributions import distributions as rsd
import torch
import torch.distributions as td
import torch.nn as nn

from abismal_torch.distributions import compute_kl_divergence
from abismal_torch.layers import *
from abismal_torch.distributions import DeltaDistribution


class ImageScaler(nn.Module):

    SUPPORTED_POSTERIORS = {
        "FoldedNormal": rsd.FoldedNormal,
        "Rice": rsd.Rice,
        "Normal": td.Normal,
        "LogNormal": td.LogNormal,
        "Gamma": td.Gamma,
        "DeltaDistribution": DeltaDistribution,
    }
    SUPPORTED_PRIORS = {
        "Cauchy": td.Cauchy,
        "Laplace": td.Laplace,
        "Normal": td.Normal,
        "HalfNormal": td.HalfNormal,
        "HalfCauchy": td.HalfCauchy,
        "Exponential": td.Exponential,
    }


    def __init__(
        self,
        mlp_width: Optional[int] = 32,
        mlp_depth: Optional[int] = 20,
        share_weights: Optional[bool] = True,
        hidden_units: Optional[int] = None,
        use_glu: Optional[bool] = False,
        activation: Optional[str] = None,
        normalization: Optional[str] = "RMSNorm",
        scaling_posterior: Optional[
            str | type[torch.distributions.Distribution]
        ] = td.Normal,
        scaling_posterior_transform: Optional[
            str | type[torch.distributions.transforms.Transform]
        ] = 'SoftplusTransform',
        scaling_kl_weight: Optional[float] = 0.01,
        scaling_prior: Optional[
            str | type[torch.distributions.Distribution]
        ] = td.Cauchy,
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
                the scaling posterior distribution. Defaults to a normal distribution.
            scaling_posterior_transform (torch.distributions.transforms.Transform, optional): transform for
                the scaling posterior distribution. Defaults to a Softplus transform. Ignored if the
                scaling posterior is a delta distribution (DeltaDistribution).
            scaling_kl_weight (float, optional): float, the weight of the KL divergence
                loss between the scaling posterior and the scaling prior. Defaults to 0.01.
                Ignored if the scaling posterior is a delta distribution.
            scaling_prior (torch.distributions.Distribution, optional): instantiated distribution
                for the scaling prior. Defaults to a Cauchy distribution with mean 0.0 and
                scale 1.0. Ignored if the scaling posterior is a delta distribution.
            scaling_prior_params (tuple, optional): tuple of parameters for the scaling prior distribution.
                Ignored if the scaling posterior is a delta distribution.
            epsilon (float, optional): float, the epsilon value for numerical stability.
                Defaults to 1e-12.
        """
        super().__init__(**kwargs)
        # Initialize the posterior, transform, prior, kl weight
        if scaling_posterior not in self.SUPPORTED_POSTERIORS.keys() and scaling_posterior not in self.SUPPORTED_POSTERIORS.values():
            raise ValueError(f"Unsupported scaling posterior: {scaling_posterior}")
        if scaling_prior not in self.SUPPORTED_PRIORS.keys() and scaling_prior not in self.SUPPORTED_PRIORS.values():
            raise ValueError(f"Unsupported scaling prior: {scaling_prior}")
        if isinstance(scaling_posterior, str):
            if scaling_posterior == "DeltaDistribution":
                self.scaling_posterior = DeltaDistribution
            elif hasattr(rsd, scaling_posterior):
                self.scaling_posterior = getattr(rsd, scaling_posterior)
            else:
                self.scaling_posterior = getattr(td, scaling_posterior)
        else:
            self.scaling_posterior = scaling_posterior
        self._num_posterior_args = len(self.scaling_posterior.arg_constraints)
        if isinstance(scaling_posterior_transform, str):
            self._transform = getattr(td.transforms, scaling_posterior_transform)
        else:
            self._transform = scaling_posterior_transform
        self.posterior_positive_transform = torch.distributions.ComposeTransform(
            [
                self._transform(),
                td.AffineTransform(epsilon, 1.0),
            ]
        )
        self.scaling_kl_weight = scaling_kl_weight
        if scaling_prior_params is None:
            self.register_buffer(
                "scaling_prior_params",
                torch.tensor([], dtype=torch.float32),
            )
        else:
            self.register_buffer(
                "scaling_prior_params",
                torch.tensor(scaling_prior_params, dtype=torch.float32),
            )
        self._scaling_prior = scaling_prior

        # Initialize architecture
        self.image_linear_in = CustomInitLazyLinear(mlp_width)
        self.scale_linear_in = CustomInitLazyLinear(mlp_width)
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
            normalization=normalization,
            epsilon=epsilon ** 0.5, # sqrt as used in std for normalization
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
                normalization=normalization,
                epsilon=epsilon ** 0.5,
            )


    def _init_scaling_prior(self) -> None:
        if isinstance(self._scaling_prior, str):
            if hasattr(rsd, self._scaling_prior):
                self.scaling_prior = getattr(rsd, self._scaling_prior)(
                    *self.scaling_prior_params
                )
            else:
                self.scaling_prior = getattr(td, self._scaling_prior)(
                *self.scaling_prior_params
            )
        else:
            self.scaling_prior = self._scaling_prior(*self.scaling_prior_params)

    def _apply_posterior_positive_transforms(
        self, raw_params: torch.Tensor
    ) -> tuple[torch.Tensor, ...]:
        """
        Enforce positivity on any posterior argument whose constraint is
        GreaterThan(0).
        """
        transformed_args: list[torch.Tensor] = []
        for idx, (arg_name, constraint) in enumerate(self.scaling_posterior.arg_constraints.items()):
            param = raw_params[..., idx]
            if (
                isinstance(constraint, td.constraints._GreaterThan)
                or isinstance(constraint, td.constraints._GreaterThanEq)
            ):
                if constraint.lower_bound == 0:
                    param = self.posterior_positive_transform(param)
            transformed_args.append(param)
        return tuple(transformed_args)

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
        # from IPython import embed
        # embed(colors="linux")
        transformed_scaling_params = self._apply_posterior_positive_transforms(scaling_params)
        q = self.scaling_posterior(*transformed_scaling_params)
        z = q.rsample(sample_shape=(mc_samples,))  # Shape (mc_samples, n_reflns)
        if self.scaling_posterior == DeltaDistribution:
            return {
                "z": torch.t(z),  # Shape (n_reflns, mc_samples)
                "kl_div": torch.zeros_like(scaling_params[..., 0]), # Shape (n_reflns,)
            }
        else:
            if not hasattr(self, "scaling_prior"):
                self._init_scaling_prior()
            p = self.scaling_prior.expand((len(scaling_params),))
            kl_div = compute_kl_divergence(q, p, samples=z) * self.scaling_kl_weight
            return {
                "z": torch.t(z),  # Shape (n_reflns, mc_samples)
                "kl_div": kl_div, # Shape scalar
            }
