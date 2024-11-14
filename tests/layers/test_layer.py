import numpy as np
import pytest
import torch
import torch.nn.init as init

from abismal_torch.layers import *
from abismal_torch.layers.feedforward import CustomInitLazyLinear


@pytest.fixture
def linear_out_size():
    return 10


@pytest.fixture
def test_params():
    return {"n_refln": 120, "n_feature": 17, "n_image": 10}


@pytest.fixture
def data(test_params):
    return torch.rand(
        (test_params["n_refln"], test_params["n_feature"]), dtype=torch.float32
    )


def test_imageaverage(test_params, data):
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


def test_lazylinear_shape(test_params, linear_out_size, data):
    linear = CustomInitLazyLinear(linear_out_size)
    out = linear(data)
    assert out.shape == (test_params["n_refln"], linear_out_size)


@pytest.mark.parametrize("seed", [42])
def test_lazylinear_init(linear_out_size, seed, data):
    # check using built-in initializers
    w_init = init.ones_
    b_init = init.zeros_
    linear = CustomInitLazyLinear(
        linear_out_size, weight_initializer=w_init, bias_initializer=b_init
    )
    _ = linear(data)
    assert torch.allclose(linear.weight, torch.ones_like(linear.weight))
    assert torch.allclose(linear.bias, torch.zeros_like(linear.bias))

    # check using custom initializer with default arguments
    w_init = VarianceScalingNormalInitializer(generator=torch.manual_seed(seed))
    b_init = init.zeros_
    linear = CustomInitLazyLinear(
        linear_out_size, weight_initializer=w_init, bias_initializer=b_init
    )
    _ = linear(data)
    alternative_weights = torch.randn_like(linear.weight)
    std = w_init.gain * np.sqrt(1.0 / w_init.fan)
    torch.nn.init.trunc_normal_(
        alternative_weights,
        0,
        std,
        -2 * std,
        2 * std,
        generator=torch.manual_seed(seed),
    )
    assert torch.allclose(linear.weight, alternative_weights)

    # check using custom initializer 2 with custom arguments
    my_activation = "leaky_relu"
    my_param = 0.2
    w_init = VarianceScalingNormalInitializer(
        mode="fan_in",
        gain=None,
        low=None,
        high=None,
        nonlinearity=my_activation,
        param=my_param,
        generator=torch.manual_seed(seed),
    )
    linear = CustomInitLazyLinear(linear_out_size, weight_initializer=w_init)
    _ = linear(data)
    alternative_weights = torch.randn_like(linear.weight)
    gain = torch.nn.init.calculate_gain(my_activation, my_param)
    std = gain * np.sqrt(1.0 / alternative_weights.T.shape[0])
    torch.nn.init.normal_(
        alternative_weights, 0, std, generator=torch.manual_seed(seed)
    )
    assert torch.allclose(linear.weight, alternative_weights)


@pytest.mark.parametrize(
    "dropout, hidden_units, activation, normalize",
    [(None, None, "ReLU", False), (0.2, 12, "SELU", True)],
)
def test_feedforward_shape(test_params, dropout, hidden_units, activation, normalize):
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
def test_mlp_shape(test_params, width, depth, hidden_width):
    mlp = MLP(width, depth, hidden_width=hidden_width)
    data = torch.randn((test_params["n_refln"], test_params["n_feature"]))
    out = mlp(data)
    assert out.shape == (test_params["n_refln"], width)
    # check depth of MLP
    assert len(list(mlp.children())) == depth + 1
    # check hidden_width
    if hidden_width is None:
        hidden_width = 2 * width
    assert mlp[1].linear1.weight.T.shape == (width, hidden_width)
