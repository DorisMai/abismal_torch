import numpy as np
import pytest
import torch

from abismal_torch.layers.standardization import (
    LazyWelfordStandardization,
    WelfordStandardization,
)


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
    batch1, batch2 = batches
    standardization = LazyWelfordStandardization(center=True)
    standardized_batch1 = standardization(batch1)
    assert torch.allclose(standardization.running_mean, batch1.mean(dim=0))
    assert torch.allclose(standardization.var, batch1.var(dim=0))
    assert standardization.num_batches_tracked == batch1.shape[0]
    assert torch.allclose(
        standardized_batch1, (batch1 - batch1.mean(dim=0)) / batch1.std(dim=0)
    )
