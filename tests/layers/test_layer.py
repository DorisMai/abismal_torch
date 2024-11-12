import numpy as np
import pytest
import torch

from abismal_torch.layers import *
from abismal_torch.layers.feedforward import VarianceScalingLazyLinear


@pytest.fixture
def test_params():
    return {"n_refln": 67, "n_feature": 12, "n_image": 10}


def test_imageaverage(test_params):
    data = torch.rand(
        test_params["n_refln"], test_params["n_feature"], dtype=torch.float32
    )
    image_id = torch.randint(0, test_params["n_image"], (test_params["n_refln"],))
    average = ImageAverage()
    out = average(data, image_id)
    assert out.shape == (test_params["n_image"], test_params["n_feature"])

    alternative_out = torch.full(
        (test_params["n_image"], test_params["n_feature"]), np.nan
    )
    for i in range(test_params["n_image"]):
        alternative_out[i] = data[image_id == i].mean(dim=0)
    assert torch.allclose(out, alternative_out)


@pytest.mark.parametrize("out_size", [10])
def test_lazylinear(test_params, out_size):
    ll = VarianceScalingLazyLinear(out_size)
    x = torch.randn((test_params["n_refln"], test_params["n_feature"]))
    out = ll(x)
    assert out.shape == (test_params["n_refln"], out_size)

    # check weights mean and std
    epsilon = 0.1
    assert ll.weight.mean().abs() < epsilon, f"Weight mean is {ll.weight.mean()}"

    fan_avg = 0.5 * (test_params["n_feature"] + out_size)
    scale = 1 / 10.0
    std = np.sqrt(scale / fan_avg)
    epsilon = std / 2
    assert (
        torch.abs(ll.weight.std() - std) < epsilon
    ), f"Weight std is {ll.weight.std()}, Expected {std}"


@pytest.mark.parametrize(
    "dropout, hidden_units, activation, normalize",
    [(None, None, "ReLU", False), (0.2, 12, "SELU", True)],
)
def test_feedforward(test_params, dropout, hidden_units, activation, normalize):
    data = torch.rand(
        test_params["n_image"],
        test_params["n_refln"],
        test_params["n_feature"],
        dtype=torch.float32,
    )
    ff = FeedForward(
        input_size=test_params["n_feature"],
        hidden_units=hidden_units,
        dropout=dropout,
        activation=activation,
        normalize=normalize,
    )
    out = ff(data)
    assert out.shape == (
        test_params["n_image"],
        test_params["n_refln"],
        test_params["n_feature"],
    )


@pytest.mark.parametrize("width, depth, hidden_width", [(4, 3, None), (16, 6, 7)])
def test_mlp(test_params, width, depth, hidden_width):
    mlp = MLP(width, depth, hidden_width=hidden_width)
    x = torch.randn((test_params["n_refln"], test_params["n_feature"]))
    out = mlp(x)
    assert out.shape == (test_params["n_refln"], width)
    # check depth of MLP
    assert len(list(mlp.children())) == depth + 1
    # check a hidden_width
    if hidden_width is None:
        hidden_width = 2 * width
    assert mlp.state_dict()["1.network.1.weight"].T.shape == (width, hidden_width)
