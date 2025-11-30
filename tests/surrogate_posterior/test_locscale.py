import gemmi
import pytest
import torch

from abismal_torch.surrogate_posterior import LocScalePosterior


@pytest.mark.parametrize("anomalous", [True, False])
@pytest.mark.parametrize("parents", [None, torch.tensor([0, 0], dtype=torch.int32)])
@pytest.mark.parametrize("rsm_distribution", ["FoldedNormal", "Rice", "Gamma"])
class TestLocScalePosterior:
    def test_locscale_posterior(self, rag, rsm_distribution):
        loc = torch.rand(rag.rac_size)
        scale = torch.rand(rag.rac_size)
        posterior = LocScalePosterior(rag, rsm_distribution, loc, scale)
        loc_param = next(posterior.parameters())
        assert loc_param.shape == loc.shape
        assert type(loc_param) == torch.nn.Parameter

    @pytest.mark.parametrize("steps", [1000])
    @pytest.mark.parametrize("mc_samples", [32])
    def test_from_unconstrained_params(
        self, rag, rsm_distribution, steps, mc_samples
    ):
        loc = torch.rand(rag.rac_size)
        scale = torch.rand(rag.rac_size)
        posterior = LocScalePosterior.from_unconstrained_params(
            rag, rsm_distribution, loc, scale
        )
        opt = torch.optim.Adam(posterior.parameters())

        prior_loc = torch.ones(rag.rac_size) * 10
        prior_scale = torch.ones(rag.rac_size) * 3
        prior = torch.distributions.Normal(prior_loc, prior_scale)

        init_kl = (
            posterior.log_prob(posterior.rsample((mc_samples,)))
            - prior.log_prob(posterior.rsample((mc_samples,)))
        ).mean()
        for _ in range(steps):
            opt.zero_grad()
            z = posterior.rsample((mc_samples,))
            kl = (posterior.log_prob(z) - prior.log_prob(z)).mean()
            kl.backward()
            opt.step()

        final_kl = (
            posterior.log_prob(posterior.rsample((mc_samples,)))
            - prior.log_prob(posterior.rsample((mc_samples,)))
        ).mean()
        assert final_kl < init_kl, f"final_kl: {final_kl}, init_kl: {init_kl}"