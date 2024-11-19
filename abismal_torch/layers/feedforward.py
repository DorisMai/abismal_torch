from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .initializers import VarianceScalingNormalInitializer


class CustomInitLazyLinear(nn.LazyLinear):
    def __init__(
        self,
        output_size: int,
        weight_initializer: Optional[nn.Module] = None,
        bias_initializer: Optional[nn.Module] = None,
        **kwargs,
    ) -> None:
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
        hidden_units: Optional[int] = None,
        activation: Optional[str | nn.Module] = "ReLU",
        dropout: Optional[float] = None,
        normalize: Optional[bool] = False,
        weight_initializer: Optional[nn.Module] = VarianceScalingNormalInitializer(),
        bias_initializer: Optional[nn.Module] = nn.init.zeros_,
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
            activation (str | nn.Module, optional): Activation function to use. If a string,
                it should be the name of a PyTorch activation function. Otherwise, it should
                be an instance of a nn.Module. Defaults to 'ReLU'.
            normalize (bool): Whether to apply layer normalization. Defaults to False.
            weight_initializer (nn.Module, optional): Weight initializer. Defaults to
                VarianceScalingNormalInitializer with default parameters.
            bias_initializer (nn.Module, optional): Bias initializer. Defaults to
                nn.init.zeros_.
        """
        super().__init__(**kwargs)
        self.input_size = input_size
        self.hidden_units = hidden_units
        if self.hidden_units is None:
            self.hidden_units = 2 * self.input_size
        if isinstance(activation, str):
            self.activation = getattr(nn.modules.activation, activation)()
        else:
            self.activation = activation
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


class FeedForward_GLU(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_units: Optional[int] = None,
        activation: Optional[str | nn.Module] = "SwiGLU",
        dropout: Optional[float] = None,
        normalize: Optional[bool] = False,
        weight_initializer: Optional[nn.Module] = VarianceScalingNormalInitializer(),
        bias_initializer: Optional[nn.Module] = None,
        **kwargs,
    ) -> None:
        """
        A feedforward layer that supports GLU activation. We follow the implementation
        in [Shazeer, 2020](https://arxiv.org/pdf/1612.08083):
        ```
        FFN_{GLU}(x, W, V, W2) = (activation(xW) ⊗ xV)W2
        ```
        where ⊗ is element-wise multiplication and biases terms are not used.

        Args:
            input_size (int): Size of input features, i.e. last dimension of input tensor.
            hidden_units (int, optional): Size of the hidden layer. Defaults to 4/3 times
                the input size to keep same number of parameters as the FFN without GLU.
            activation (str | nn.Module, optional): Activation function to use. If a string,
                it should be one of the following: 'SwiGLU' (default), 'GEGLU', 'ReGLU',
                'GLU', or 'Bilinear'. Otherwise, it should be an instance of a nn.Module
                that takes two tensors as input.
            dropout (float, optional): Dropout rate to apply after the second linear layer.
                Defaults to None.
            normalize (bool): Whether to apply layer normalization. Defaults to False.
            weight_initializer (nn.Module, optional): Weight initializer. Defaults to
                VarianceScalingNormalInitializer with default parameters.
            bias_initializer (nn.Module, optional): Bias initializer. Defaults to None.
        """
        super().__init__(**kwargs)
        self.activation = activation
        self.dropout = dropout
        self.normalize = normalize
        self.input_size = input_size
        self.hidden_units = hidden_units
        if self.hidden_units is None:
            self.hidden_units = int(4 / 3 * self.input_size)

        linear_params = {
            "weight_initializer": weight_initializer,
            "bias_initializer": bias_initializer,
            "bias": False,
        }
        self.W = CustomInitLazyLinear(self.hidden_units, **linear_params)
        self.V = CustomInitLazyLinear(self.hidden_units, **linear_params)
        self.W2 = CustomInitLazyLinear(self.input_size, **linear_params)

    def _get_activation(self, h1: torch.Tensor, h2: torch.Tensor) -> torch.Tensor:
        """
        Get the activation function for the GLU.

        Args:
            h1 (torch.Tensor): comes from xW, shape (batch_size, hidden_units)
            h2 (torch.Tensor): comes from xV, shape (batch_size, hidden_units)

        Returns:
            torch.Tensor: Activated tensor of shape (batch_size, hidden_units).
        """
        if isinstance(self.activation, nn.Module):
            return self.activation(h1, h2)

        if self.activation == "SwiGLU":
            return F.silu(h1) * h2
        elif self.activation == "GEGLU":
            return F.gelu(h1) * h2
        elif self.activation == "ReGLU":
            return F.relu(h1) * h2
        elif self.activation == "GLU":
            return F.sigmoid(h1) * h2
        elif self.activation == "Bilinear":
            return h1 * h2
        else:
            raise ValueError(f"Activation function {self.activation} not supported")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.normalize:
            x = F.layer_norm(x, (self.input_size,))
        h1 = self.W(x)
        h2 = self.V(x)
        out = self.W2(self._get_activation(h1, h2))
        if self.dropout is not None:
            out = F.dropout(out, self.dropout)
        return out


class MLP(nn.Sequential):
    def __init__(
        self,
        width: int,
        depth: int,
        input_layer: Optional[bool] = False,
        use_glu: Optional[bool] = False,
        **kwargs,
    ) -> None:
        """
        A Multi-Layer Perceptron (MLP) with depth sets of feedforward layers.

        Args:
            width (int): Width of the input and output embedding.
            depth (int): Number of feedforward modules.
            input_layer (bool): Whether to have an input linear layer at the beginning.
            use_glu (bool): Whether to use GLU activation in the feedforward modules.
            **kwargs: Keyword arguments for FeedForward.
        """
        layers = []
        if input_layer:
            layers.append(CustomInitLazyLinear(width))
        if use_glu:
            for _ in range(depth):
                layers.append(FeedForward_GLU(width, **kwargs))
        else:
            for _ in range(depth):
                layers.append(FeedForward(width, **kwargs))
        super().__init__(*layers)
