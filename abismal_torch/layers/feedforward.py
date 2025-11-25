from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .initializers import VarianceScalingNormalInitializer

NORMALIZER_DICT = {
    "RMSNorm": lambda s, x: x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + s.epsilon),
    "LayerNorm": lambda s, x: (x - x.mean(-1, keepdim=True)) / (x.std(-1, keepdim=True) + s.epsilon),
    "activation": lambda s, x: s.activation(x),
    "identity": lambda s, x: x,
}

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
        activation: Optional[str] = "ReLU",
        dropout: Optional[float] = None,
        normalization: Optional[str] = "RMSNorm",
        weight_initializer: Optional[nn.Module] = VarianceScalingNormalInitializer(),
        bias_initializer: Optional[nn.Module] = nn.init.zeros_,
        epsilon: Optional[float] = 1e-6,
        **kwargs,
    ) -> None:
        """
        This is a ResNet version 2 style feedforward layer. It implements the following
        ```
        out = dropout(linear(activation(hidden_linear(activation or norm(in)))))) + in
        ```
        Where dropout and layer normalization are optional, and the linear layers are
        initialized with variance scaling.

        Args:
            input_size (int): Size of input features, i.e. last dimension of input tensor.
            dropout (float, optional): Dropout rate to apply after the second linear layer.
                Defaults to None.
            hidden_units (int, optional): Size of the hidden layer. Defaults to 2 times
                the input size.
            activation (str | nn.Module): Activation function to use. If a string,
                it should be the name of a PyTorch activation function. Otherwise, it should
                be an instance of a nn.Module. Defaults to 'ReLU'.
            normalization (str, optional): Normalization function to use. Only 'RMSNorm' and 
                'LayerNorm' are supported. You can also replace with "activation" to use the 
                same activation function or "identity" to skip normalization. Defaults to 'RMSNorm'.
            weight_initializer (nn.Module, optional): Weight initializer. Defaults to
                VarianceScalingNormalInitializer with default parameters.
            bias_initializer (nn.Module, optional): Bias initializer. Defaults to
                nn.init.zeros_.
            epsilon (float, optional): Epsilon value for the normalization functions. Defaults to
                1e-6.
        """
        super().__init__(**kwargs)
        self.input_size = input_size
        self.hidden_units = hidden_units
        if self.hidden_units is None:
            self.hidden_units = 2 * self.input_size
        self.activation = activation or "ReLU"
        self.activation = getattr(nn.modules.activation, self.activation)()
        self.normalization = normalization or "identity"
        self.epsilon = epsilon
        self.dropout = dropout
        self.linear1 = CustomInitLazyLinear(
            self.hidden_units, weight_initializer, bias_initializer
        )
        self.linear2 = CustomInitLazyLinear(
            self.input_size, weight_initializer, bias_initializer
        )

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        return NORMALIZER_DICT[self.normalization](self, x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_in = x
        out = self.linear1(self.normalize(x))
        out = self.linear2(self.activation(out))
        if self.dropout is not None:
            out = F.dropout(out, self.dropout)
        out = out + x_in
        return out


class FeedForward_GLU(nn.Module):

    ACTIVATION_DICT = {
        "SwiGLU": lambda h1, h2: F.silu(h1) * h2,
        "GEGLU": lambda h1, h2: F.gelu(h1) * h2,
        "ReGLU": lambda h1, h2: F.relu(h1) * h2,
        "GLU": lambda h1, h2: F.sigmoid(h1) * h2,
        "Bilinear": lambda h1, h2: h1 * h2,
    }

    def __init__(
        self,
        input_size: int,
        hidden_units: Optional[int] = None,
        activation: Optional[str] = "SwiGLU",
        dropout: Optional[float] = None,
        normalization: Optional[str] = "RMSNorm",
        weight_initializer: Optional[nn.Module] = VarianceScalingNormalInitializer(),
        bias_initializer: Optional[nn.Module] = None,
        epsilon: Optional[float] = 1e-6,
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
            activation (str): Activation function to use. Only supports one of the following:
                'SwiGLU' (default), 'GEGLU', 'ReGLU', 'GLU', or 'Bilinear'.
            dropout (float, optional): Dropout rate to apply after the second linear layer.
                Defaults to None.
            normalization (str): Normalization function to use. Only supports 'RMSNorm' and 
                'LayerNorm'. You can also replace with "activation" to use the same activation
                function or "identity" to skip normalization. Defaults to 'RMSNorm'.
            weight_initializer (nn.Module, optional): Weight initializer. Defaults to
                VarianceScalingNormalInitializer with default parameters.
            bias_initializer (nn.Module, optional): Bias initializer. Defaults to None.
        """
        super().__init__(**kwargs)
        self.activation = activation or "SwiGLU"
        self.dropout = dropout
        self.normalization = normalization or "identity"
        self.input_size = input_size
        self.hidden_units = hidden_units
        if self.hidden_units is None:
            self.hidden_units = int(4 / 3 * self.input_size)
        self.epsilon = epsilon
        linear_params = {
            "weight_initializer": weight_initializer,
            "bias_initializer": bias_initializer,
            "bias": False,
        }
        self.W = CustomInitLazyLinear(self.hidden_units, **linear_params)
        self.V = CustomInitLazyLinear(self.hidden_units, **linear_params)
        self.W2 = CustomInitLazyLinear(self.input_size, **linear_params)

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        return NORMALIZER_DICT[self.normalization](self, x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_in = x
        h = self.normalize(x)
        h1 = self.W(h)
        h2 = self.V(h)
        out = self.W2(self.ACTIVATION_DICT[self.activation](h1, h2))
        if self.dropout is not None:
            out = F.dropout(out, self.dropout)
        out = out + x_in
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
