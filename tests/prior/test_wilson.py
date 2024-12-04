import gemmi
import pytest

from abismal_torch.prior.structure_factor.wilson import *
from abismal_torch.symmetry import ReciprocalASU, ReciprocalASUGraph


@pytest.mark.parametrize("anomalous", [True, False])
@pytest.mark.parametrize("parents", [None, torch.tensor([0, 0], dtype=torch.int32)])
class TestWilson:
    @pytest.fixture
    def rasu_params(self):
        return {
            "spacegroups": [gemmi.SpaceGroup(19), gemmi.SpaceGroup(4)],
            "dmins": [5.1, 3.8],
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

    @pytest.fixture
    def z(self, rag, mc_samples=10):
        return torch.distributions.Exponential(rate=1).rsample(
            (mc_samples, rag.rac_size)
        )

    def test_single_wilson(self, rag, z):
        centric = rag.centric
        multiplicity = rag.multiplicity
        normalization_sigma = torch.rand(rag.rac_size) + 0.1
        single_wilson = GenericWilsonDistribution(
            centric, multiplicity, normalization_sigma
        )

        assert single_wilson.log_prob(z).shape == z.shape

        alternative_p_centric = torch.distributions.HalfNormal(
            torch.sqrt(multiplicity * normalization_sigma)
        )
        alternative_p_acentric = torch.distributions.Weibull(
            torch.sqrt(multiplicity * normalization_sigma), 2
        )
        log_probs = torch.where(
            centric,
            alternative_p_centric.log_prob(z),
            alternative_p_acentric.log_prob(z),
        )
        assert torch.allclose(single_wilson.log_prob(z), log_probs, atol=1e-5)

    def test_multi_wilson(self, rag, z):
        centric = rag.centric
        multiplicity = rag.multiplicity
        normalization_sigma = torch.rand(rag.rac_size) + 0.1
        correlation = torch.rand(rag.rac_size)
        single_wilson = GenericWilsonDistribution(
            centric, multiplicity, normalization_sigma
        )
        multi_wilson = GenericWilsonDistribution(
            centric,
            multiplicity,
            normalization_sigma,
            single_wilson=False,
            correlation=correlation,
        )
        is_root = torch.ones(rag.rac_size, dtype=torch.bool)
        parents_reflection_ids = torch.arange(rag.rac_size)
        multi_wilson = MultiWilsonDistribution(
            is_root,
            parents_reflection_ids,
            centric,
            multiplicity,
            normalization_sigma,
            correlation,
        )
        assert multi_wilson.log_prob(z).shape == z.shape
        assert torch.allclose(
            multi_wilson.log_prob(z), single_wilson.log_prob(z), atol=1e-5
        )

    def test_wilson_prior(self, rag, z, custom_params):
        single_wilson_prior = WilsonPrior(rag)
        default_distribution = single_wilson_prior.distribution()
        assert default_distribution.log_prob(z).shape == z.shape

        custom_id, custom_rasu_id, custom_hkl = custom_params
        custom_distribution = single_wilson_prior.distribution(
            custom_rasu_id, custom_hkl
        )
        assert (
            custom_distribution.log_prob(z[:, custom_id]).shape == z[:, custom_id].shape
        )

    def test_multi_wilson_prior(self, rag, z, custom_params, correlation=[0, 0.9]):
        multi_wilson_prior = MultiWilsonPrior(rag, correlation=correlation)
        default_distribution = multi_wilson_prior.distribution()
        assert default_distribution.log_prob(z).shape == z.shape

        custom_id, custom_rasu_id, custom_hkl = custom_params
        custom_distribution = multi_wilson_prior.distribution(
            custom_rasu_id, custom_hkl
        )
        assert (
            custom_distribution.log_prob(z[:, custom_id]).shape == z[:, custom_id].shape
        )
