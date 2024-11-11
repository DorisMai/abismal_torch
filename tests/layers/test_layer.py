import numpy as np
import pytest
import torch

from abismal_torch.layers import FeedForward


@pytest.mark.parametrize(
    "dropout, hidden_units, activation, xavier_gain, normalize",
    [(None, None, "ReLU", None, False), (0.2, 12, "SELU", 0.8, True)],
)
def test_resnet(dropout, hidden_units, activation, xavier_gain, normalize):
    n_image = 10
    n_refln = 100
    n_feature = 5
    data = torch.rand(n_image, n_refln, n_feature, dtype=torch.float32)
    ff = FeedForward(
        input_size=n_feature,
        hidden_units=hidden_units,
        dropout=dropout,
        activation=activation,
        xavier_gain=xavier_gain,
        normalize=normalize,
    )
    out = ff(data)
    assert out.shape == (n_image, n_refln, n_feature)
