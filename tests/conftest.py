import gemmi
import pytest
import torch

from abismal_torch.symmetry import ReciprocalASU, ReciprocalASUGraph
from os.path import abspath, dirname, join


@pytest.fixture
def rasu_params():
    return {
        "spacegroups": [gemmi.SpaceGroup(19), gemmi.SpaceGroup(4)],
        "dmins": [9.1, 8.8],
        "cell": gemmi.UnitCell(10.0, 20.0, 30.0, 90.0, 90.0, 90.0),
    }


@pytest.fixture
def rag(rasu_params, anomalous, parents):
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
def custom_params(rag):
    custom_n_reflections = int(rag.rac_size * 1.2)
    custom_id = torch.randint(0, rag.rac_size, (custom_n_reflections,))
    custom_rasu_id = rag.rasu_ids[custom_id]
    custom_hkl = rag.H_rasu[custom_id]
    return custom_id, custom_rasu_id, custom_hkl


@pytest.fixture
def data_params():
    return {"n_refln": 91, "n_feature": 18, "n_image": 7}


@pytest.fixture
def custom_scaling_model_params():
    return {
        "mlp_width": 20,
        "mlp_depth": 4,
        "hidden_units": 15,
        "activation": None,
    }

@pytest.fixture
def refl_file():
    """
    Path to a dials stills reflection table
    """
    datapath = ["..", "data", "dials.refl"]
    file_name = abspath(join(dirname(__file__), *datapath))
    return file_name

@pytest.fixture
def expt_file():
    """
    Path to a dials stills experiment list
    """
    datapath = ["..", "data", "dials.expt"]
    file_name = abspath(join(dirname(__file__), *datapath))
    return file_name


