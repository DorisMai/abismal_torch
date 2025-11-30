import pytest
import torch

from abismal_torch.likelihood import StudentTLikelihood
from abismal_torch.merging import VariationalMergingModel
from abismal_torch.prior.structure_factor.wilson import WilsonPrior
from abismal_torch.scaling import ImageScaler
from abismal_torch.surrogate_posterior import FoldedNormalPosterior


@pytest.mark.parametrize("anomalous", [True, False])
@pytest.mark.parametrize("parents", [None, torch.tensor([0, 0], dtype=torch.int32)])
class TestMerging:
    @pytest.fixture
    def scaling_model(self, custom_scaling_model_params):
        return ImageScaler(
            share_weights=True, use_glu=True, **custom_scaling_model_params
        )

    @pytest.fixture
    def wilson_prior(self, rag):
        return WilsonPrior(rag)

    @pytest.fixture
    def sf_posterior(self, rag, wilson_prior):
        loc_init = wilson_prior.distribution().mean
        scale_init = 0.01 * loc_init
        return FoldedNormalPosterior.from_unconstrained_loc_and_scale(
            rag, loc_init, scale_init
        )

    @pytest.fixture
    def student_likelihood(self):
        return StudentTLikelihood(dof=16)

    @pytest.fixture
    def merging_model(
        self, scaling_model, sf_posterior, wilson_prior, student_likelihood
    ):
        return VariationalMergingModel(
            scale_model=scaling_model,
            surrogate_posterior=sf_posterior,
            prior=wilson_prior,
            likelihood=student_likelihood,
            mc_samples=8,
        )

    @pytest.fixture
    def myinputs(self, rag, data_params):
        n_refln = data_params["n_refln"]
        n_feature = data_params["n_feature"]
        n_image = data_params["n_image"]

        valid_indices = torch.nonzero(rag.reflection_id_grid != -1)
        idx = torch.randint(0, len(valid_indices), (n_refln,))
        sampled_indices = valid_indices[idx]

        image_id = torch.randint(0, n_image, (n_refln,))
        asu_id = sampled_indices[:, 0]
        hkl_in = sampled_indices[:, 1:]
        resolution = torch.rand(n_refln, 1)
        wavelength = torch.ones(n_refln, 1)
        metadata = torch.rand(n_refln, n_feature)
        iobs = torch.randn(n_refln, 1)
        sigiobs = torch.rand(n_refln, 1)

        return {
            "image_id": torch.randint(0, n_image, (n_refln,)),
            "rasu_id": sampled_indices[:, 0],
            "hkl_in": sampled_indices[:, 1:],
            "resolution": torch.rand(n_refln, 1),
            "wavelength": torch.ones(n_refln, 1),
            "metadata": torch.rand(n_refln, n_feature),
            "iobs": torch.randn(n_refln, 1),
            "sigiobs": torch.rand(n_refln, 1),
        }

    @pytest.mark.parametrize("steps", [100])
    def test_merging(self, merging_model, myinputs, steps, data_params, rag):
        xout = merging_model(myinputs)
        assert xout["ipred_avg"].shape == (
            myinputs["iobs"].shape[0],
        ), f"xout[ipred_aveg] shape is {xout['ipred_avg'].shape}"
        assert xout["loss_nll"].shape == torch.Size(
            [data_params["n_image"]]
        ), f"xout[loss_nll].shape is {xout['loss_nll'].shape}"
        assert torch.numel(xout["loss_kl"]) == 1, f"xout[loss_kl] is not a scalar, shape: {xout['loss_kl'].shape}"

        opt = torch.optim.Adam(merging_model.parameters())
        for _ in range(steps):
            opt.zero_grad()
            xout = merging_model(myinputs)
            loss = xout["loss_nll"].mean() + xout["loss_kl"].mean()
            loss.backward()
            opt.step()
