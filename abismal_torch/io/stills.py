import json
from typing import Iterable, List, Optional, Sequence

import gemmi
import numpy as np
import reciprocalspaceship as rs
import torch
from reciprocalspaceship.decorators import cellify, spacegroupify

from abismal_torch.io.dataset import AbismalDataset


class StillsDataset(AbismalDataset):
    __DEFAULT_METADATA_KEYS__ = [
        "dHKL",
        "delpsical.rad",
        "s1.0",
        "s1.1",
        "s1.2",
        "xyzcal.px.0",
        "xyzcal.px.1",
        #'xyzcal.px.2',
    ]
    __HANDLER_TYPE__ = "dials"

    @cellify
    @spacegroupify
    def __init__(
        self,
        expt_file: str,
        refl_file: str,
        cell: Optional[Sequence[float]] = None,
        spacegroup: Optional[str] = None,
        wavelength: Optional[float] = None,
        rasu_id: Optional[int] = 0,
        dmin: Optional[float] = 0.0,
        dials_metadata_keys: Optional[List[str]] = None,
        **kwargs,
    ):
        """
        Custom Pytorch Dataset Class for DIALS Data

        Args:
            expt_file (str): A DIALS experiment (.expt) file name
            refl_file (str): A DIALS reflection (.refl) file name
            cell (list[float], optional): a list of cell parameters. If provided, overrides the cell parameters in the input file.
            spacegroup (str, optional): a spacegroup symbol. If provided, overrides the spacegroup in the input file.
            wavelength (float, optional): a wavelength. By default this will be read from the .expt files.
            rasu_id (int, optional): a rasu id. Defaults to 0.
            dmin (float, optional): Highest resolution to include. Defaults to 0, all reflections are included.
        """
        super().__init__()
        self._cell = cell
        self.spacegroup = spacegroup
        self.wavelength = wavelength
        self.rasu_id = rasu_id
        self.dmin = dmin
        self.refl_file = refl_file
        self.expt_file = expt_file
        if dials_metadata_keys is None:
            self.metadata_keys = self.__DEFAULT_METADATA_KEYS__
        else:
            self.metadata_keys = dials_metadata_keys

    @staticmethod
    def _can_handle(input_files):
        for f in input_files:
            if f.split(".")[-1] not in ("expt", "refl"):
                return False
        return True

    @property
    def refl_file(self):
        return self._refl_file

    @refl_file.setter
    def refl_file(self, refl_file):
        self._refl_file = refl_file
        self.reset()

    @property
    def expt_file(self):
        return self._expt_file

    @expt_file.setter
    def expt_file(self, expt_file):
        self._expt_file = expt_file
        self.reset()
        if self.cell is None:
            self.cell = self.get_average_cell(expt_file)
        if self.spacegroup is None:
            self.spacegroup = self.get_space_group(expt_file)

    @staticmethod
    def real_space_axes_to_cell_params(
        real_a: Sequence[float],
        real_b: Sequence[float],
        real_c: Sequence[float],
    ):
        a, b, c = map(np.linalg.norm, (real_a, real_b, real_c))
        alpha = rs.utils.angle_between(real_b, real_c)
        beta = rs.utils.angle_between(real_a, real_c)
        gamma = rs.utils.angle_between(real_a, real_b)
        for x in (90.0, 120.0):
            if np.isclose(alpha, x):
                alpha = np.round(alpha)
            if np.isclose(beta, x):
                alpha = np.round(beta)
            if np.isclose(gamma, x):
                alpha = np.round(gamma)
        return [a, b, c, alpha, beta, gamma]

    @staticmethod
    def get_average_cell(expt_file: str) -> gemmi.UnitCell:
        import json

        with open(expt_file) as f:
            js = json.load(f)
        crystals = js["crystal"]
        cells = []
        for i, exp in enumerate(js["experiment"]):
            cid = exp["crystal"]
            crystal = js["crystal"][cid]
            cells.append(
                StillsDataset.real_space_axes_to_cell_params(
                    crystal["real_space_a"],
                    crystal["real_space_b"],
                    crystal["real_space_c"],
                )
            )
        cell_params = np.array(cells).mean(0)
        return gemmi.UnitCell(*cell_params)

    @staticmethod
    def get_space_group(expt_file: str, check_consistent=False) -> gemmi.SpaceGroup:
        import json

        with open(expt_file) as f:
            js = json.load(f)

        sg = None
        for i, exp in enumerate(js["experiment"]):
            cid = exp["crystal"]
            crystal = js["crystal"][cid]
            hall = crystal["space_group_hall_symbol"]
            go = gemmi.symops_from_hall(hall)
            _sg = gemmi.find_spacegroup_by_ops(go)
            if sg is None:
                sg = _sg
            if not check_consistent:
                return _sg
            if sg != _sg:
                raise ValueError(
                    f"Expected experiment {i} to have spacegroup {sg} but it has {_sg}."
                )
        return sg

    def _load_tensor_data(self):
        ds = (
            rs.io.read_dials_stills(
                self.refl_file,
                unitcell=self.cell,
                spacegroup=self.spacegroup,
            )
            .compute_dHKL()
            .label_absences()
        )

        # Compute wavelength sans experiment list
        ds["wavelength"] = np.reciprocal(
            np.linalg.norm(ds[["s1.0", "s1.1", "s1.2"]], axis=-1)
        )
        ds["wavelength"] = ds.groupby("BATCH")["wavelength"].transform("mean")

        # Remove systematic absences & and apply resolution cutoff
        ds = ds[(~ds["ABSENT"]) & (ds["dHKL"] >= self.dmin)]

        batch = torch.tensor(ds["BATCH"].to_numpy(), dtype=torch.int32)
        rasu_id = self.rasu_id * torch.ones(len(ds), dtype=batch.dtype)
        hkl = torch.tensor(ds[["H", "K", "L"]].to_numpy("int32"))
        d = torch.tensor(ds["dHKL"].to_numpy("float32"))
        wavelength = torch.tensor(ds["wavelength"].to_numpy("float32"))
        metadata = torch.tensor(ds[self.metadata_keys].to_numpy("float32"))
        iobs = torch.tensor(ds["intensity.sum.value"].to_numpy("float32"))
        sigiobs = torch.tensor(
            np.sqrt(ds["intensity.sum.variance"].to_numpy("float32"))
        )

        self._tensor_data = {
            "image_id": batch,
            "rasu_id": rasu_id,
            "hkl_in": hkl,
            "resolution": d,
            "wavelength": wavelength,
            "metadata": metadata,
            "iobs": iobs,
            "sigiobs": sigiobs,
        }

    def __len__(self) -> int:
        """
        Expensive, fallback implementation of __len__. If possible overload this to avoid
        calling _load_tensor_data.
        """
        if self._length is None:
            with open(self.expt_file) as f:
                js = json.load(f)
            self._length = len(js["experiment"])
        return self._length

    @classmethod
    def from_sequence(
        cls,
        input_files: Iterable[str],
        dmin: List[float] | float = 0.0,
        wavelength: List[float] | float = 1.0,
        rasu_id: List[int] | int = 0,
        cell: List[gemmi.UnitCell] | gemmi.UnitCell = None,
        spacegroup: List[gemmi.SpaceGroup] | gemmi.SpaceGroup = None,
        **kwargs,
    ) -> List[AbismalDataset]:

        kwargs.update(
            {
                "refl_file": [f for f in input_files if f.endswith(".refl")],
                "expt_file": [f for f in input_files if f.endswith(".expt")],
                "dmin": dmin,
                "wavelength": wavelength,
                "rasu_id": rasu_id,
                "cell": cell,
                "spacegroup": spacegroup,
            }
        )
        if len(kwargs["refl_file"]) != len(kwargs["expt_file"]):
            raise ValueError("Number of refl_files and expt_files must be the same")
        length = len(kwargs["refl_file"])
        for k, v in kwargs.items():
            expand = True
            expand &= (not isinstance(v, list)) and (not isinstance(v, tuple))
            expand |= k == "cell"
            expand |= "metadata_keys" in k
            if expand:
                kwargs[k] = [v] * length
            elif len(v) != length:
                raise ValueError(
                    f"Length of {k} must be the same as the number of input files"
                )
        result = []
        for i in range(length):
            result.append(cls(**{k: v[i] for k, v in kwargs.items()}))
        return result
