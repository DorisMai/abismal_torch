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
        max_nrefln_per_image: Optional[int] = 300,
    ):
        """
        Custom Pytorch Dataset Class for MTZ files.

        Args:
            mtz_file (str): a path to the MTZ file.
            cell (list[float], optional): a list of cell parameters. If provided, overrides the cell parameters in the MTZ file.
            spacegroup (str, optional): a spacegroup symbol. If provided, overrides the spacegroup in the MTZ file.
            batch_key (str, optional): a key for the batch or image. If not provided, the first column of type B is used.
            intensity_key (str, optional): a key for the intensity. If not provided, the first column of type J is used.
            sigma_key (str, optional): a key for the intensity uncertainty. If not provided, the first column of type Q is used.
            metadata_keys (list[str], optional): a list of metadata keys. Defaults to ["XDET", "YDET"].
            wavelength (float, optional): a wavelength. Defaults to 1.0.
            rasu_id (int, optional): a rasu id. Defaults to 0.
            dmin (float, optional): Highest resolution to include.
            max_nrefln_per_image (int, optional): Maximum number of reflections per image to include in a training batch. This is
                used to avoid loading too many reflections into memory at once. Defaults to 300.
        """
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
            self.cell = ds.cell.parameters
        if self.spacegroup is None:
            self.spacegroup = ds.spacegroup.hm
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

        self.image_id = torch.tensor(ds.image_id.to_numpy("int32"))
        self.rasu_id = torch.ones_like(self.image_id, dtype=torch.int32) * self.rasu_id
        self.hkl_in = torch.tensor(ds.get_hkls())
        self.resolution = torch.tensor(ds.dHKL.to_numpy())
        self.wavelength = (
            torch.ones_like(self.image_id, dtype=torch.float32) * self.wavelength
        )
        self.metadata = torch.tensor(ds[self.metadata_keys].to_numpy())
        self.Iobs = torch.tensor(ds[self.intensity_key].to_numpy())
        self.SigIobs = torch.tensor(ds[self.sigma_key].to_numpy())
        # Create image_id to indices mapping for fast lookup
        self.imageid_to_dataid = {}
        for img_id in range(self.image_id.max() + 1):
            self.imageid_to_dataid[img_id] = torch.where(self.image_id == img_id)[0]
        self.max_nrefln_per_image = max_nrefln_per_image

    def __len__(self):
        return self.image_id.max() + 1

    def __getitem__(self, idx):
        indices = self.imageid_to_dataid[idx]
        if len(indices) > self.max_nrefln_per_image:
            # randomly draw max_nrefln_per_image reflections from indices
            indices = torch.randperm(len(indices))[:self.max_nrefln_per_image]
        return {
            "image_id": self.image_id[indices],
            "rasu_id": self.rasu_id[indices],
            "hkl_in": self.hkl_in[indices],
            "resolution": self.resolution[indices],
            "wavelength": self.wavelength[indices],
            "metadata": self.metadata[indices],
            "iobs": self.Iobs[indices],
            "sigiobs": self.SigIobs[indices],
        }

    def _get_first_key_of_type(self, ds, dtype):
        idx = ds.dtypes == dtype
        if idx.sum() == 0:
            raise ValueError(f"Dataset has no key of type {dtype}")
        key = ds.dtypes[idx].keys()[0]
        return key
