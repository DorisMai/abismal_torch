import gemmi
import numpy as np
import pytest
import torch
from reciprocalspaceship.utils import (
    compute_structurefactor_multiplicity,
    generate_reciprocal_cell,
    hkl_to_asu,
    is_absent,
    is_centric,
)

from abismal_torch.symmetry import (
    ReciprocalASU,
    ReciprocalASUCollection,
    ReciprocalASUGraph,
)


@pytest.mark.parametrize("anomalous", [True, False])
@pytest.mark.parametrize("spacegroup", [gemmi.SpaceGroup(19)])
@pytest.mark.parametrize("cell", [gemmi.UnitCell(10.0, 20.0, 30.0, 90.0, 90.0, 90.0)])
@pytest.mark.parametrize("dmin", [10.0])
class TestReciprocalASU:
    @pytest.fixture
    def rasu_data(self, cell, spacegroup, dmin, anomalous):
        rasu = ReciprocalASU(cell, spacegroup, dmin, anomalous)
        return rasu

    @pytest.fixture
    def Hcell(self, rasu_data):
        h, k, l = rasu_data.Hmax
        H = np.mgrid[-h : h + 1 : 1, -k : k + 1 : 1, -l : l + 1 : 1].reshape((3, -1)).T

        # Apply resolution cutoff and remove absences
        d = rasu_data.cell.calculate_d_array(H)
        H = H[d >= rasu_data.dmin]
        H = H[~is_absent(H, rasu_data.spacegroup)]
        # Remove 0, 0, 0
        H = H[np.any(H != 0, axis=1)]

        return H

    @pytest.fixture
    def alternative_rasu_data(self, Hcell, spacegroup, anomalous):
        Hasu, Isym = hkl_to_asu(Hcell, spacegroup)
        go = spacegroup.operations()
        if anomalous:
            friedel_sign = np.array([-1, 1])[Isym % 2][:, None]
            centric = go.centric_flag_array(Hasu).astype(bool)
            friedel_sign[centric] = 1
            Hasu = friedel_sign * Hasu

        Hasu, inv = np.unique(Hasu, axis=0, return_inverse=True)
        return {
            "Hasu": Hasu,
            "centric": is_centric(Hasu, spacegroup).astype(bool),
            "multiplicity": compute_structurefactor_multiplicity(
                Hasu, spacegroup
            ).astype(np.float32),
            "reflection_id_1D": np.arange(inv.max() + 1)[inv].astype(np.int32),
        }

    def test_rasu_attributes(self, rasu_data, alternative_rasu_data, Hcell):
        assert np.array_equal(rasu_data.H_rasu, alternative_rasu_data["Hasu"]), "Hasu"
        assert np.array_equal(
            rasu_data.centric.numpy(), alternative_rasu_data["centric"]
        ), "centric"
        assert np.array_equal(
            rasu_data.multiplicity.numpy(), alternative_rasu_data["multiplicity"]
        ), "multiplicity"
        reflection_id_1D = rasu_data.reflection_id_grid[
            Hcell[:, 0], Hcell[:, 1], Hcell[:, 2]
        ].numpy()
        assert np.array_equal(
            reflection_id_1D, alternative_rasu_data["reflection_id_1D"]
        ), "reflection_id"

    # def test_rasu_gather(self, rasu_data, Hcell):
    #     # Generate random image assignment for each reflection
    #     n_images = 10
    #     image_id = torch.tensor(
    #         np.sort(np.random.choice(n_images, len(Hcell))).astype(np.int32)
    #     )

    #     # Create nested Hobs tensor based on image assignment
    #     unique_image_id, counts = torch.unique(image_id, return_counts=True)
    #     offsets = torch.zeros_like(image_id, dtype=torch.bool)
    #     offsets[torch.cumsum(counts[:-1], 0)] = True
    #     segment_starts = torch.nonzero(offsets).squeeze(1)
    #     nested_Hobs = torch.tensor_split(torch.tensor(Hcell), segment_starts)
    #     nested_Hobs = list(nested_Hobs)

    #     # Test source array using reflection ID
    #     source = torch.arange(rasu_data.rasu_size)
    #     gathered = rasu_data.gather(source, nested_Hobs)

    #     assert gathered.size(0) == len(unique_image_id)
    #     flattened_gathered = torch.cat([t for t in gathered], dim=0)
    #     assert torch.equal(
    #         flattened_gathered,
    #         source[rasu_data.reflection_id_grid[Hcell[:, 0], Hcell[:, 1], Hcell[:, 2]]],
    #     )


@pytest.mark.parametrize("anomalous", [True, False])
@pytest.mark.parametrize("spacegroups", [[gemmi.SpaceGroup(19), gemmi.SpaceGroup(4)]])
@pytest.mark.parametrize("cell", [gemmi.UnitCell(10.0, 20.0, 30.0, 90.0, 90.0, 90.0)])
@pytest.mark.parametrize("dmins", [[10.0, 8.6]])
class TestReciprocalASUCollectionandGraph:
    @pytest.fixture
    def rac_data(self, cell, anomalous, spacegroups, dmins):
        rac = []
        for spacegroup, dmin in zip(spacegroups, dmins):
            rac.append(ReciprocalASU(cell, spacegroup, dmin, anomalous))
        return rac

    @pytest.mark.parametrize("n_reflection", [20])
    def test_rac_gather(self, rac_data, n_reflection):
        rac = ReciprocalASUCollection(*rac_data)
        H = np.zeros((n_reflection, 3), dtype=np.int32)
        reflection_ids = torch.zeros(n_reflection, dtype=torch.int32)

        # Randomly assign n_reflection number of reflections across len(rac_data) rasus
        rasu_ids = torch.randint(len(rac), (n_reflection,))
        n_refln_per_rasu = torch.bincount(rasu_ids, minlength=len(rac))
        for rasu_id, n_refln in enumerate(n_refln_per_rasu):
            rasu = rac.reciprocal_asus[rasu_id]
            H_rasu = rasu.H_rasu
            reflection_id_rasu = rasu.gather(
                torch.arange(rasu.rasu_size, dtype=torch.int32), torch.tensor(H_rasu)
            )
            # Randomly get n_refln number of valid reflections from current rasu
            chosen_ids = torch.randint(len(H_rasu), (n_refln,))
            H[rasu_ids == rasu_id] = H_rasu[chosen_ids]
            reflection_ids[rasu_ids == rasu_id] = reflection_id_rasu[chosen_ids]

        source = torch.cat(
            [torch.arange(rasu.rasu_size) for rasu in rac.reciprocal_asus]
        )
        gathered = rac.gather(source, rasu_ids, torch.tensor(H))
        assert torch.equal(gathered, reflection_ids)
