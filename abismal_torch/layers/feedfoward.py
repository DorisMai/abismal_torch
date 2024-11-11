from typing import Optional

import lightning as L
import torch
import torch.nn as nn


class FeedForward(L.LightningModule):
    def __init__(
        self,
        input_size: int,
        dropout: Optional[float] = None,
        hidden_units: Optional[int] = None,
        activation: Optional[str] = "ReLU",
        xavier_gain: Optional[float] = None,
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
        self.xavier_gain = xavier_gain
        if self.xavier_gain is None:
            self.xavier_gain = nn.init.calculate_gain(activation.lower())
        self.activation = getattr(nn.modules.activation, activation)()
        if self.hidden_units is None:
            self.hidden_units = 2 * self.input_size
        self.linear1 = nn.Linear(self.input_size, self.hidden_units)
        self.linear2 = nn.Linear(self.hidden_units, self.input_size)

        self.network = nn.Sequential()
        if normalize:
            self.network.append(nn.LayerNorm(input_size))
        self.network.append(self.activation)
        self.network.append(self._xavier_initialize(self.linear1))
        self.network.append(self.activation)
        self.network.append(self._xavier_initialize(self.linear2))
        if dropout is not None:
            self.network.append(nn.Dropout(dropout))

    def _xavier_initialize(self, linear: nn.Linear) -> nn.Linear:
        """
        Initialize the weights of a linear layer using Xavier initialization.

        Args:
            linear (nn.Linear): Linear layer to initialize.

        Returns:
            nn.Linear: The initialized linear layer.
        """
        nn.init.xavier_normal_(linear.weight, gain=self.xavier_gain)
        nn.init.zeros_(linear.bias)
        return linear

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_in = x
        x = self.network(x)
        return x + x_in
