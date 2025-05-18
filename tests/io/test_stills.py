import pytest
from abismal_torch.io.stills import StillsDataset


def test_stills_dataset(refl_file, expt_file):
    ds = StillsDataset(refl_file, expt_file)

