from typing import Optional

import torch
import torch.nn as nn

from .initializers import VarianceScalingNormalInitializer


class CustomInitLazyLinear(nn.LazyLinear):
    def __init__(
        self,
        output_size: int,
        weight_initializer: Optional[nn.Module] = None,
        bias_initializer: Optional[nn.Module] = None,
        **kwargs,
    ):
        """
        A lazy linear layer with custom weight and bias initialization. The
        initializers can be the built-in methods from torch.nn.init, or a custom one.

        Args:
            output_size (int): Size of the output features.
            weight_initializer (nn.Module, optional): Weight initializer.
            bias_initializer (nn.Module, optional): Bias initializer.
        """
        super().__init__(output_size, **kwargs)
        self.weight_initializer = weight_initializer
        self.bias_initializer = bias_initializer

    def reset_parameters(self) -> None:
        if self.in_features == 0:
            return
        if self.weight_initializer is None:
            super().reset_parameters()
        else:
            self.weight_initializer(self.weight)
        if self.bias is not None and self.bias_initializer is not None:
            self.bias_initializer(self.bias)


class FeedForward(nn.Module):
    def __init__(
        self,
        input_size: int,
        dropout: Optional[float] = None,
        hidden_units: Optional[int] = None,
        activation: Optional[str] = "ReLU",
        normalize: Optional[bool] = False,
        weight_initializer: Optional[nn.Module] = None,
        bias_initializer: Optional[nn.Module] = None,
        **kwargs,
    ) -> None:
        """
        This is a ResNet version 2 style feedforward layer. It implements the following
        ```
        out = dropout(linear(activation(hidden_linear(activation(layer_norm(in)))))) + in
        ```
        Where dropout and layer normalization are optional, and the linear layers are
        initialized with variance scaling.

        Args:
            input_size (int): Size of input features, i.e. last dimension of input tensor.
            dropout (float, optional): Dropout rate to apply after the second linear layer.
                Defaults to None.
            hidden_units (int, optional): Size of the hidden layer. Defaults to 2 times
                the input size.
            activation (str): Name of PyTorch activation function to use. Defaults to 'ReLU'.
            normalize (bool): Whether to apply layer normalization. Defaults to False.
            weight_initializer (nn.Module, optional): Weight initializer.
            bias_initializer (nn.Module, optional): Bias initializer.
        """
        super().__init__(**kwargs)
        self.input_size = input_size
        self.hidden_units = hidden_units
        self.activation = getattr(nn.modules.activation, activation)()
        if self.hidden_units is None:
            self.hidden_units = 2 * self.input_size
        self.linear1 = CustomInitLazyLinear(
            self.hidden_units, weight_initializer, bias_initializer
        )
        self.linear2 = CustomInitLazyLinear(
            self.input_size, weight_initializer, bias_initializer
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


class MLP(nn.Sequential):
    def __init__(
        self,
        width: int,
        depth: int,
        hidden_width: Optional[int] = None,
        input_layer: Optional[bool] = True,
        weight_initializer: Optional[nn.Module] = VarianceScalingNormalInitializer(),
        bias_initializer: Optional[nn.Module] = nn.init.zeros_,
    ):
        """
        A Multi-Layer Perceptron (MLP) with depth sets of feedforward layers.

        Args:
            width (int): Width of the input and output layers.
            depth (int): Number of feedforward modules.
            hidden_width (int, optional): Width of the hidden layers. Defaults to None,
                which FeedForward will set it to 2 times the width.
            input_layer (bool, optional): Whether to include a Linear layer at the
                beginning of the MLP. Defaults to True.
            weight_initializer (nn.Module, optional): Weight initializer for the FeedForward module.
                Defaults to VarianceScalingNormalInitializer().
            bias_initializer (nn.Module, optional): Bias initializer for the FeedForward module.
                Defaults to nn.init.zeros_.
        """
        layers = []
        if input_layer:
            layers.append(CustomInitLazyLinear(width))
        for i in range(depth):
            layers.append(
                FeedForward(
                    width,
                    hidden_units=hidden_width,
                    weight_initializer=weight_initializer,
                    bias_initializer=bias_initializer,
                )
            )
        super().__init__(*layers)
