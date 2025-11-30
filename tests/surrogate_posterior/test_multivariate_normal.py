import gemmi
import pytest
import torch

from abismal_torch.surrogate_posterior import MultivariateNormalPosterior


@pytest.mark.parametrize("anomalous", [True, False])
@pytest.mark.parametrize("parents", [None, torch.tensor([0, 0], dtype=torch.int32)])
@pytest.mark.parametrize("rank", [1, 5])
class TestMultivariateNormalPosterior:
    def test_multivariate_normal_posterior(self, rag, rank):
        loc = torch.rand(rag.rac_size)
        cov_factor = torch.rand(rag.rac_size, rank)
        cov_diag = torch.rand(rag.rac_size)
        posterior = MultivariateNormalPosterior(rag, loc, cov_factor, cov_diag)
        loc_param = next(posterior.parameters())
        assert loc_param.shape == loc.shape
        assert type(loc_param) == torch.nn.Parameter

    @pytest.mark.parametrize("steps", [1000])
    @pytest.mark.parametrize("mc_samples", [32])
    def test_multivariate_normal_posterior_from_unconstrained_loc_and_scale(
        self, rag, rank, steps, mc_samples,
    ):
        loc = torch.rand(rag.rac_size)
        cov_factor = torch.rand(rag.rac_size, rank)
        cov_diag = torch.rand(rag.rac_size)
        posterior = MultivariateNormalPosterior.from_unconstrained_loc_and_scale(
            rag, loc, cov_factor, cov_diag
        )
        opt = torch.optim.Adam(posterior.parameters())

        prior_loc = torch.ones(rag.rac_size) * 10
        prior_scale = torch.ones(rag.rac_size) * 3
        prior = torch.distributions.Independent(
            torch.distributions.Normal(prior_loc, prior_scale), 
            reinterpreted_batch_ndims=1
        )

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