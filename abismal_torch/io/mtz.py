from typing import List, Optional

import reciprocalspaceship as rs
from abismal_torch.io.dataset import AbismalDataset
import torch
from torch.utils.data import Dataset


class MTZDataset(AbismalDataset):
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
        dmin: Optional[float] = 0.,
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
        """
        super().__init__(
            cell=cell,
            spacegroup=spacegroup,
            wavelength=wavelength,
            rasu_id=rasu_id,
            dmin=dmin,
        )
        self.mtz_file = mtz_file
        self.batch_key = batch_key
        self.intensity_key = intensity_key
        self.sigma_key = sigma_key
        self.wavelength = wavelength
        self.metadata_keys = metadata_keys

    @staticmethod
    def _can_handle(input_files):
        for f in input_files:
            if not f.endswith(".mtz"):
                return False
        return True

    @property
    def mtz_file(self):
        return self._mtz_file

    @mtz_file.setter
    def mtz_file(self, mtz_file):
        self.reset()
        self._mtz_file = mtz_file

    def _load_tensor_data(self):
        ds = rs.read_mtz(self.mtz_file)

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
        ds = ds[(ds.dHKL >= self.dmin) & (~ds.ABSENT)]

        ds["image_id"] = ds.groupby(self.batch_key).ngroup()
        ds.sort_values("image_id", inplace=True)

        self._tensor_data = {
            "image_id" : torch.tensor(ds.image_id.to_numpy("int32")),
            "rasu_id" : torch.ones(len(ds), dtype=torch.int32) * self.rasu_id,
            "hkl_in" : torch.tensor(ds.get_hkls()),
            "resolution" : torch.tensor(ds.dHKL.to_numpy()),
            "wavelength" : torch.ones(len(ds), dtype=torch.float32) * self.wavelength,
            "metadata" : torch.tensor(ds[self.metadata_keys].to_numpy()),
            "iobs" : torch.tensor(ds[self.intensity_key].to_numpy()), 
            "sigiobs" : torch.tensor(ds[self.sigma_key].to_numpy()),
        }


    def _get_first_key_of_type(self, ds, dtype):
        idx = ds.dtypes == dtype
        if idx.sum() == 0:
            raise ValueError(f"Dataset has no key of type {dtype}")
        key = ds.dtypes[idx].keys()[0]
        return key
