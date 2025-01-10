from typing import List, Optional

import reciprocalspaceship as rs
import torch
from torch.utils.data import Dataset


class MTZDataset(Dataset):
    def __init__(
        self,
        mtz_file: str,
        cell: Optional[List[float]] = None,
        spacegroup: Optional[str] = None,
        batch_key: Optional[str] = None,
        intensity_key: Optional[str] = None,
        sigma_key: Optional[str] = None,
        metadata_keys: Optional[List[str]] = None,
        wavelength: Optional[float] = None,
        rasu_id: Optional[int] = 0,
        dmin: Optional[float] = None,
    ):
        ds = rs.read_mtz(mtz_file)
        self.cell = cell
        self.spacegroup = spacegroup
        self.wavelength = wavelength
        self.rasu_id = rasu_id
        self.dmin = dmin
        self.batch_key = batch_key
        self.intensity_key = intensity_key
        self.sigma_key = sigma_key
        self.metadata_keys = metadata_keys

        if self.cell is None:
            self.cell = ds.cell
        if self.spacegroup is None:
            self.spacegroup = ds.spacegroup
        if self.wavelength is None:
            self.wavelength = 1.0
        if self.batch_key is None:
            self.batch_key = self._get_first_key_of_type(ds, "B")
        if self.intensity_key is None:
            self.intensity_key = self._get_first_key_of_type(ds, "J")
        if self.sigma_key is None:
            self.sigma_key = self._get_first_key_of_type(ds, "Q")
        if self.metadata_keys is None:
            self.metadata_keys = [
                "XDET",
                "YDET",
            ]

        ds.compute_dHKL(True).label_absences(True)
        ds = ds[~ds.ABSENT]
        if self.dmin is not None:
            ds = ds[ds.dHKL >= self.dmin]
        ds["image_id"] = ds.groupby(self.batch_key).ngroup()
        ds.sort_values("image_id", inplace=True)

        self.image_id = torch.tensor(ds.image_id.to_numpy("int64"))
        self.rasu_id = torch.ones_like(self.image_id, dtype=torch.int32) * self.rasu_id
        self.hkl_in = torch.tensor(ds.get_hkls())
        self.resolution = torch.tensor(ds.dHKL.to_numpy())
        self.wavelength = (
            torch.ones_like(self.image_id, dtype=torch.float32) * self.wavelength
        )
        self.metadata = torch.tensor(ds[self.metadata_keys].to_numpy())
        self.Iobs = torch.tensor(ds[self.intensity_key].to_numpy())
        self.SigIobs = torch.tensor(ds[self.sigma_key].to_numpy())

    def __len__(self):
        return len(self.Iobs)

    def __getitem__(self, idx):
        return (
            self.image_id[idx],
            self.rasu_id[idx],
            self.hkl_in[idx],
            self.resolution[idx],
            self.wavelength[idx],
            self.metadata[idx],
            self.Iobs[idx],
            self.SigIobs[idx],
        )

    def _get_first_key_of_type(self, ds, dtype):
        idx = ds.dtypes == dtype
        if idx.sum() == 0:
            raise ValueError(f"Dataset has no key of type {dtype}")
        key = ds.dtypes[idx].keys()[0]
        return key
