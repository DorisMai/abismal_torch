import numpy as np
import pytest
import torch
import torch.nn.init as init

from abismal_torch.layers import *
from abismal_torch.layers.feedforward import CustomInitLazyLinear, FeedForward_GLU
from abismal_torch.scaling import ImageScaler


@pytest.fixture
def linear_out_size():
    return 10


@pytest.fixture
def data(data_params):
    return torch.rand(
        (data_params["n_refln"], data_params["n_feature"]), dtype=torch.float32
    )


@pytest.fixture
def image_id(data_params):
    return torch.randint(0, data_params["n_image"], (data_params["n_refln"],))


def test_imageaverage(data_params, data, image_id):
    average = ImageAverage()
    out, _, _ = average(data, image_id)
    assert out.shape == (data_params["n_image"], data_params["n_feature"])

    alternative_out = torch.full(
        (data_params["n_image"], data_params["n_feature"]), np.nan
    )
    for i in range(data_params["n_image"]):
        alternative_out[i] = data[image_id == i].mean(dim=0)
    assert torch.allclose(out, alternative_out)


def test_lazylinear_shape(data_params, linear_out_size, data):
    linear = CustomInitLazyLinear(linear_out_size)
    out = linear(data)
    assert out.shape == (data_params["n_refln"], linear_out_size)


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
def test_feedforward_shape(data_params, dropout, hidden_units, activation, normalize):
    data = torch.rand(
        data_params["n_image"],
        data_params["n_refln"],
        data_params["n_feature"],
        dtype=torch.float32,
    )
    ff = FeedForward(
        input_size=data_params["n_feature"],
        hidden_units=hidden_units,
        dropout=dropout,
        activation=activation,
        normalize=normalize,
    )
    out = ff(data)
    assert out.shape == (
        data_params["n_image"],
        data_params["n_refln"],
        data_params["n_feature"],
    )


def test_feedforward_glu_shape(data_params, data):
    ff_glu = FeedForward_GLU(data_params["n_feature"])
    out = ff_glu(data)
    assert out.shape == (data_params["n_refln"], data_params["n_feature"])
    # check if the number of parameters is the same as FFN without GLU
    ff_nweights = 4 * data_params["n_feature"] ** 2
    ff_glu_nweights = sum(p.numel() for p in ff_glu.parameters())
    assert (
        ff_nweights == ff_glu_nweights
    ), "Default FF and FF_GLU number of weights are {} and {}".format(
        ff_nweights, ff_glu_nweights
    )


@pytest.mark.parametrize("hidden_units", [None, 7])
def test_mlp_shape(data_params, hidden_units):
    custom_params = {
        "width": 16,
        "depth": 6,
        "input_layer": True,
        "dropout": 0.1,
        "normalize": True,
        "use_glu": False,
        "activation": "GELU",
        "weight_initializer": torch.nn.init.ones_,
    }
    mlp = MLP(hidden_units=hidden_units, **custom_params)
    data = torch.randn((data_params["n_refln"], data_params["n_feature"]))
    out = mlp(data)
    assert out.shape == (data_params["n_refln"], custom_params["width"])
    # check depth of MLP
    assert len(list(mlp.children())) == custom_params["depth"] + 1
    # check hidden_units
    if hidden_units is None:
        hidden_units = 2 * custom_params["width"]
    assert mlp[1].linear1.weight.T.shape == (custom_params["width"], hidden_units)


@pytest.mark.parametrize("share_weights", [True, False])
@pytest.mark.parametrize("use_glu", [True, False])
def test_image_scaler(
    data_params, custom_scaling_model_params, data, image_id, share_weights, use_glu
):
    metadata = data
    iobs = torch.randn(data_params["n_refln"], 1)
    sigiobs = torch.randn(data_params["n_refln"], 1)
    inputs = {
        "metadata": metadata,
        "iobs": iobs,
        "sigiobs": sigiobs,
    }
    mc_samples = 8

    custom_scaling_model = ImageScaler(
        share_weights=share_weights, use_glu=use_glu, **custom_scaling_model_params
    )
    _ = custom_scaling_model(inputs, image_id, mc_samples)

    if share_weights:
        # image_linear_in, scale_linear_in, linear_out, pool, mlp
        assert len(list(custom_scaling_model.children())) == 5
    else:
        # image_linear_in, scale_linear_in, linear_out, pool, mlp, scale_mlp
        assert len(list(custom_scaling_model.children())) == 6

    if use_glu:
        assert custom_scaling_model.mlp[1].W.weight.T.shape == (
            custom_scaling_model_params["mlp_width"],
            custom_scaling_model_params["hidden_units"],
        )
        if share_weights:
            mlp_w = custom_scaling_model.mlp[-1].W2.weight
            scale_mlp_w = custom_scaling_model.scale_mlp[-1].W2.weight
            assert torch.allclose(mlp_w, scale_mlp_w)
    else:
        assert custom_scaling_model.mlp[1].linear1.weight.T.shape == (
            custom_scaling_model_params["mlp_width"],
            custom_scaling_model_params["hidden_units"],
        )
        if share_weights:
            mlp_w = custom_scaling_model.mlp[-1].linear2.weight
            scale_mlp_w = custom_scaling_model.scale_mlp[-1].linear2.weight
            assert torch.allclose(mlp_w, scale_mlp_w)
