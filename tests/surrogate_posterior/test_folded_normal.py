import gemmi
import pytest
import torch
from rs_distributions import distributions as rsd

from abismal_torch.surrogate_posterior import FoldedNormalPosterior
from abismal_torch.symmetry import ReciprocalASU, ReciprocalASUGraph


@pytest.mark.parametrize("anomalous", [True, False])
@pytest.mark.parametrize("parents", [None, torch.tensor([0, 0], dtype=torch.int32)])
class TestFoldedNormalPosterior:
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

        init_kl = (posterior.log_prob(posterior.rsample((mc_samples,))) - prior.log_prob(posterior.rsample((mc_samples,)))).mean()
        for _ in range(steps):
            opt.zero_grad()
            z = posterior.rsample((mc_samples,))
            kl = (posterior.log_prob(z) - prior.log_prob(z)).mean()
            kl.backward()
            opt.step()

        final_kl = (posterior.log_prob(posterior.rsample((mc_samples,))) - prior.log_prob(posterior.rsample((mc_samples,)))).mean()
        assert final_kl < init_kl, f"final_kl: {final_kl}, init_kl: {init_kl}"
