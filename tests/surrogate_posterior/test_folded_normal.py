import gemmi
import pytest
import torch
from rs_distributions import distributions as rsd

from abismal_torch.surrogate_posterior import FoldedNormalPosterior
from abismal_torch.symmetry import ReciprocalASU, ReciprocalASUGraph


@pytest.mark.parametrize("anomalous", [True, False])
@pytest.mark.parametrize("parents", [None, torch.tensor([0, 0], dtype=torch.int32)])
class TestFoldedNormalPosterior:
    @pytest.fixture
    def rasu_params(self):
        return {
            "spacegroups": [gemmi.SpaceGroup(19), gemmi.SpaceGroup(4)],
            "dmins": [9.1, 8.8],
            "cell": gemmi.UnitCell(10.0, 20.0, 30.0, 90.0, 90.0, 90.0),
        }

    @pytest.fixture
    def rag(self, rasu_params, anomalous, parents):
        rasu1 = ReciprocalASU(
            rasu_params["cell"],
            rasu_params["spacegroups"][0],
            rasu_params["dmins"][0],
            anomalous,
        )
        rasu2 = ReciprocalASU(
            rasu_params["cell"],
            rasu_params["spacegroups"][1],
            rasu_params["dmins"][1],
            anomalous,
        )
        rag = ReciprocalASUGraph(rasu1, rasu2, parents=parents)
        return rag

    @pytest.fixture
    def custom_params(self, rag):
        custom_n_reflections = int(rag.rac_size * 1.2)
        custom_id = torch.randint(0, rag.rac_size, (custom_n_reflections,))
        custom_rasu_id = rag.rasu_ids[custom_id]
        custom_hkl = rag.H_rasu[custom_id]
        return custom_id, custom_rasu_id, custom_hkl

    def test_folded_normal_posterior(self, rag):
        loc = torch.rand(rag.rac_size)
        scale = torch.rand(rag.rac_size)
        posterior = FoldedNormalPosterior(rag, loc, scale)
        loc_param = next(posterior.parameters())
        assert loc_param.shape == loc.shape
        assert type(loc_param) == torch.nn.Parameter

    @pytest.mark.parametrize("steps", [5000])
    @pytest.mark.parametrize("mc_samples", [128])
    def test_folded_normal_posterior_from_unconstrained_loc_and_scale(
        self, rag, steps, mc_samples
    ):
        loc = torch.rand(rag.rac_size)
        scale = torch.rand(rag.rac_size)
        posterior = FoldedNormalPosterior.from_unconstrained_loc_and_scale(
            rag, loc, scale
        )
        opt = torch.optim.Adam(posterior.parameters())

        prior_loc = torch.ones(rag.rac_size) * 10
        prior_scale = torch.ones(rag.rac_size) * 3
        prior = rsd.FoldedNormal(prior_loc, prior_scale)

        loc_diff_before = (posterior.distribution.loc - prior.loc).abs().mean()
        scale_diff_before = (posterior.distribution.scale - prior.scale).abs().mean()

        for _ in range(steps):
            opt.zero_grad()
            z = posterior.rsample((mc_samples,))
            kl = (posterior.log_prob(z) - prior.log_prob(z)).mean()
            kl.backward()
            opt.step()

        loc_diff_after = (posterior.distribution.loc - prior.loc).abs().mean()
        scale_diff_after = (posterior.distribution.scale - prior.scale).abs().mean()

        assert loc_diff_after < loc_diff_before
        assert scale_diff_after < scale_diff_before
