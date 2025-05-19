import pytest
from abismal_torch.io.stills import StillsDataset


def test_stills_dataset(refl_file, expt_file):
    length = 3
    ds = StillsDataset(refl_file, expt_file)
    assert len(ds) == length
    for batch in ds:
        pass

    ds[0]
    ds[length - 1]
    ds[-length]
    
