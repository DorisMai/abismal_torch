import math
from typing import Optional

import torch
import torch.nn as nn


class VarianceScalingLazyLinear(nn.LazyLinear):
    def reset_parameters(self) -> None:
        if self.in_features == 0:
            return
        fan_in, fan_out = torch.nn.init._calculate_fan_in_and_fan_out(self.weight)
        fan_avg = 0.5 * (fan_in + fan_out)
        scale = 1 / 10.0
        std = math.sqrt(scale / fan_avg)
        mean = 0.0
        low, high = -2 * std, 2 * std
        torch.nn.init.trunc_normal_(self.weight, mean, std, low, high)
        if self.bias is not None:
            torch.nn.init.zeros_(self.bias)


class FeedForward(nn.Module):
    def __init__(
        self,
        input_size: int,
        dropout: Optional[float] = None,
        hidden_units: Optional[int] = None,
        activation: Optional[str] = "ReLU",
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
        self.linear1 = VarianceScalingLazyLinear(self.hidden_units)
        self.linear2 = VarianceScalingLazyLinear(self.input_size)

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
