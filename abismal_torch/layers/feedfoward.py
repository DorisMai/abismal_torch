from typing import Optional

import torch
import torch.nn as nn


class VarianceScalingLinear(nn.Linear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: Optional[bool] = True,
        mode: Optional[str] = "fan_avg",
        scale: Optional[float] = 1.0,
        distribution: Optional[str] = "truncated_normal",
        low: Optional[float] = -2,
        high: Optional[float] = 2,
    ) -> None:
        """
        Linear layer with variance scaling initialization using truncated normal distribution.

        Args:
            in_features (int): Size of input features
            out_features (int): Size of output features
            bias (bool): If True, adds a learnable bias to the output
            mode (str): One of 'fan_in', 'fan_out', or 'fan_avg'
            scale (float): Scaling factor for the weights
            distribution (str): 'truncated_normal' or 'normal'
            low (float): Lower bound for truncated normal distribution
            high (float): Upper bound for truncated normal distribution
        """
        super().__init__(in_features, out_features, bias)
        self.initialize_parameters(mode, scale, distribution, low, high)

    def initialize_parameters(self, mode, scale, distribution, low, high):
        fan_in = self.in_features
        fan_out = self.out_features

        if mode == "fan_in":
            fan = fan_in
        elif mode == "fan_out":
            fan = fan_out
        elif mode == "fan_avg":
            fan = (fan_in + fan_out) / 2
        else:
            raise ValueError(f"Invalid mode: {mode}")

        std = torch.sqrt(torch.tensor(scale / fan))

        if distribution == "truncated_normal":
            nn.init.trunc_normal_(
                self.weight, mean=0, std=std, a=low * std, b=high * std
            )
        elif distribution == "normal":
            nn.init.normal_(self.weight, std=std)
        else:
            raise ValueError(f"Invalid distribution: {distribution}")

        if self.bias is not None:
            nn.init.zeros_(self.bias)


class FeedForward(nn.Module):
    def __init__(
        self,
        input_size: int,
        dropout: Optional[float] = None,
        hidden_units: Optional[int] = None,
        activation: Optional[str] = "ReLU",
        kernel_init_scale: Optional[float] = 1.0,
        kernel_init_mode: Optional[str] = "fan_avg",
        kernel_init_distribution: Optional[str] = "truncated_normal",
        normalize: Optional[bool] = False,
        **kwargs,
    ) -> None:
        """
        This is a ResNet version 2 style feedforward layer. It implements the following
        ```
        out = dropout(linear(activation(hidden_linear(activation(layer_norm(in)))))) + in
        ```
        Where dropout and layer normalization are optional.

        Args:
            input_size (int): Size of input features, i.e. last dimension of input tensor.
            dropout (float, optional): Dropout rate to apply after the second linear layer.
                Defaults to None.
            hidden_units (int, optional): Size of the hidden layer. Defaults to 2 times
                the input size.
            activation (str): Name of PyTorch activation function to use. Defaults to 'ReLU'.
            xavier_gain (float, optional): Gain for Xavier initialization. Defaults to None,
                in which case the gain is calculated from ReLu using `nn.init.calculate_gain`
                (note that the name required by this function can be annoyingly inconsistent
                with the arg `activation` needed to look up the activation function).
            normalize (bool): Whether to apply layer normalization. Defaults to False.
        """
        super().__init__(**kwargs)
        self.input_size = input_size
        self.hidden_units = hidden_units
        self.activation = getattr(nn.modules.activation, activation)()
        if self.hidden_units is None:
            self.hidden_units = 2 * self.input_size
        self.linear1 = VarianceScalingLinear(
            self.input_size, self.hidden_units, scale=kernel_init_scale
        )
        self.linear2 = VarianceScalingLinear(
            self.hidden_units, self.input_size, scale=kernel_init_scale
        )

        self.network = nn.Sequential()
        if normalize:
            self.network.append(nn.LayerNorm(input_size))
        self.network.append(self.activation)
        self.network.append(self.linear1)
        self.network.append(self.activation)
        self.network.append(self.linear2)
        if dropout is not None:
            self.network.append(nn.Dropout(dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_in = x
        x = self.network(x)
        return x + x_in
