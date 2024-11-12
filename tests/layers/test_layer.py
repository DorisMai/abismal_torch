import numpy as np
import pytest
import torch

from abismal_torch.layers import FeedForward


@pytest.mark.parametrize(
    "dropout, hidden_units, activation, kernel_init_scale, normalize",
    [(None, None, "ReLU", 1.0, False), (0.2, 12, "SELU", 0.1, True)],
)
def test_feedforward(dropout, hidden_units, activation, kernel_init_scale, normalize):
    n_image = 10
    n_refln = 100
    n_feature = 5
    data = torch.rand(n_image, n_refln, n_feature, dtype=torch.float32)
    ff = FeedForward(
        input_size=n_feature,
        hidden_units=hidden_units,
        dropout=dropout,
        activation=activation,
        kernel_init_scale=kernel_init_scale,
        normalize=normalize,
    )
    out = ff(data)
    assert out.shape == (n_image, n_refln, n_feature)
