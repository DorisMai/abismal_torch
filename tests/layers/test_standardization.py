import numpy as np
import pytest
import torch
import torch.distributions as td

from abismal_torch.layers.standardization import (LazyMovingStandardization,
                                                  LazyWelfordStandardization,
                                                  MovingStandardization,
                                                  WelfordStandardization)


@pytest.fixture
def batches(data_params, batch_size=10):
    batch1 = torch.rand((batch_size, data_params["n_feature"]), dtype=torch.float32)
    batch2 = torch.rand((batch_size, data_params["n_feature"]), dtype=torch.float32)
    return batch1, batch2


def test_welford_standardization(batches):
    batch1, batch2 = batches
    standardization = WelfordStandardization(num_features=batch1.shape[1], center=False)
    standardized_batch1 = standardization(batch1)
    assert torch.allclose(standardization.running_mean, batch1.mean(dim=0))
    assert torch.allclose(standardization.var, batch1.var(dim=0))
    assert standardization.num_batches_tracked == batch1.shape[0]
    assert torch.allclose(standardized_batch1, batch1 / batch1.std(dim=0))

    _ = standardization(batch2)
    ref_mean = torch.cat([batch1, batch2]).mean(dim=0)
    ref_var = torch.cat([batch1, batch2]).var(dim=0)
    assert torch.allclose(standardization.running_mean, ref_mean)
    assert torch.allclose(standardization.var, ref_var)
    assert standardization.num_batches_tracked == batch1.shape[0] + batch2.shape[0]


def test_lazy_welford_standardization(batches):
    batch1, _ = batches
    standardization = LazyWelfordStandardization(center=True)
    standardized_batch1 = standardization(batch1)
    assert torch.allclose(standardization.running_mean, batch1.mean(dim=0))
    assert torch.allclose(standardization.var, batch1.var(dim=0))
    assert standardization.num_batches_tracked == batch1.shape[0]
    assert torch.allclose(
        standardized_batch1, (batch1 - batch1.mean(dim=0)) / batch1.std(dim=0)
    )


@pytest.fixture
def data_distribution(ndim=5, distribution_type=td.Normal):
    return distribution_type(
        torch.arange(1, ndim + 1, dtype=torch.float32),
        torch.arange(1, ndim + 1, dtype=torch.float32),
    )


@pytest.mark.parametrize("center", [True, False])
@pytest.mark.parametrize("decay", [0.99, 0.999])
@pytest.mark.parametrize("batch_size", [1, 120])
def test_lazy_moving_standardization(
    data_distribution, center, decay, batch_size, nsamples=100_000
):
    samples = data_distribution.sample((nsamples,))
    standardization = LazyMovingStandardization(decay=decay, center=center)
    for i in range(0, nsamples, batch_size):
        batch = samples[i : i + batch_size]
        standardized_batch = standardization(batch)

    assert torch.allclose(standardization.mean, data_distribution.mean, rtol=0.5)
    assert torch.allclose(standardization.var, data_distribution.variance, rtol=0.5)
    assert standardization.num_batches_tracked == int(np.ceil(nsamples / batch_size))
    if batch_size > 1:
        assert torch.allclose(
            standardized_batch.std(dim=0), torch.ones(batch.shape[1]), atol=1
        )

    standardized_batch_mean = standardized_batch.mean(dim=0)
    if center:
        expected_standardized_batch = (
            (batch - data_distribution.mean) / data_distribution.stddev
        ).mean(dim=0)
        assert torch.allclose(
            standardized_batch_mean, expected_standardized_batch, atol=1
        ), f"batch mean: {batch.mean(dim=0)}, standardized batch mean: {standardized_batch_mean}, expected standardized batch mean: {expected_standardized_batch}"
    else:
        expected_standardized_batch = (batch / data_distribution.stddev).mean(dim=0)
        assert torch.allclose(
            standardized_batch_mean, expected_standardized_batch, atol=1
        ), f"batch mean: {batch.mean(dim=0)}, standardized batch mean: {standardized_batch_mean}, expected standardized batch mean: {expected_standardized_batch}"
