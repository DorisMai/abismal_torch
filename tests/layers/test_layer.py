import numpy as np
import pytest
import torch
import torch.nn.init as init

from abismal_torch.layers import *
from abismal_torch.layers.feedforward import CustomInitLazyLinear
from abismal_torch.scaling import ImageScaler


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


@pytest.fixture
def image_id(test_params):
    return torch.randint(0, test_params["n_image"], (test_params["n_refln"],))


def test_imageaverage(test_params, data, image_id):
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


@pytest.mark.parametrize("width, depth, hidden_units", [(4, 3, None), (16, 6, 7)])
def test_mlp_shape(test_params, width, depth, hidden_units):
    mlp = MLP(width, depth, input_layer=True, hidden_units=hidden_units)
    data = torch.randn((test_params["n_refln"], test_params["n_feature"]))
    out = mlp(data)
    assert out.shape == (test_params["n_refln"], width)
    # check depth of MLP
    assert len(list(mlp.children())) == depth + 1
    # check hidden_units
    if hidden_units is None:
        hidden_units = 2 * width
    assert mlp[1].linear1.weight.T.shape == (width, hidden_units)


@pytest.mark.parametrize("share_weights", [True])
def test_image_scaler(test_params, data, image_id, share_weights):
    metadata = data
    iobs = torch.randn(test_params["n_refln"], 1)
    sigiobs = torch.randn(test_params["n_refln"], 1)
    inputs = (metadata, iobs, sigiobs)
    mc_samples = 8
    basic_scaling_model = ImageScaler(share_weights=share_weights)
    output = basic_scaling_model(inputs, image_id, mc_samples)
    assert output.shape == (test_params["n_refln"], mc_samples)
    if share_weights:
        mlp_w = basic_scaling_model.mlp[-1].linear2.weight
        scale_mlp_w = basic_scaling_model.scale_mlp[-1].linear2.weight
        assert torch.allclose(mlp_w, scale_mlp_w)

    custom_params = {
        "mlp_width": 20,
        "mlp_depth": 4,
        "hidden_units": 15,
        "activation": "ReLU",
    }
    fancy_scaling_model = ImageScaler(share_weights=share_weights, **custom_params)
    output = fancy_scaling_model(inputs, image_id, mc_samples)
    # check width and depth
    assert len(list(fancy_scaling_model.children())) == custom_params["mlp_depth"] + 1
    assert fancy_scaling_model.mlp[1].linear1.weight.T.shape == (
        custom_params["mlp_width"],
        custom_params["hidden_units"],
    )
