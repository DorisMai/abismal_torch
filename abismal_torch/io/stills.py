from typing import List, Optional, Sequence

import reciprocalspaceship as rs
from reciprocalspaceship.decorators import spacegroupify,cellify
import torch
from torch.utils.data import Dataset,IterableDataset
import gemmi
import numpy as np
from abismal_torch.io.dataset import AbismalDataset
import json


class StillsDataset(AbismalDataset):
    @cellify
    @spacegroupify
    def __init__(
        self,
        refl_file: str,
        expt_file: str,
        cell: Optional[Sequence[float]] = None,
        spacegroup: Optional[str] = None,
        wavelength: Optional[float] = None,
        rasu_id: Optional[int] = 0,
        dmin: Optional[float] = 0.,
    ):
        """
        Custom Pytorch Dataset Class for DIALS Data

        Args:
            refl_file (str): A DIALS reflection (.refl) file name
            cell (list[float], optional): a list of cell parameters. If provided, overrides the cell parameters in the MTZ file.
            spacegroup (str, optional): a spacegroup symbol. If provided, overrides the spacegroup in the MTZ file.
            wavelength (float, optional): a wavelength. By default this will be read from the .expt files. 
            rasu_id (int, optional): a rasu id. Defaults to 0.
            dmin (float, optional): Highest resolution to include.
            expt_file (str): A DIALS experiment (.expt) file name
        """
        super().__init__()
        self._cell = cell
        self.spacegroup = spacegroup
        self.wavelength = wavelength
        self.rasu_id = rasu_id
        self.dmin = dmin
        self.expt_file = expt_file
        self.refl_file = refl_file
        if len(expt_files) != len(refl_files):
            raise ValueError("len(expt_files) is not the same as len(refl_files)")

        # Try to be smart and use the inputs to infer the symmetry 
        if self.cell is None:
            self.cell = StillsDataset.get_average_cell(self.expt_file)
        if self.spacegroup is None:
            self.spacegroup = StillsDataset.get_space_group([self.expt_file])

    @staticmethod
    def _can_handle(input_files):
        for f in input_files:
            if f.split('.')[-1] not in ("expt", "refl"):
                return False
        return True

    @property
    def refl_file(self):
        return self._refl_file

    @refl_file.setter
    def refl_file(self, refl_file):
        if refl_file != self._refl_file:
            self._refl_file = refl_file
            self.reset()

    @property
    def expt_file(self, expt_file):
        return self._expt_file

    @expt_file.setter
    def expt_file(self, expt_file):
        self.elist = ExperimentListFactory.from_json_file(expt_file, check_format=False)
        self._expt_file = expt_file
        self.reset()
        if self._cell is None:
            self.get_average_cell(elist)
        if self._space_group is None:
            self.get_space_group(elist)

    @staticmethod
    def real_space_axes_to_cell_params(
        real_a : Sequence[float], 
        real_b : Sequence[float], 
        real_c Sequence[float]
    ):
        a,b,c = map(np.linalg.norm, (real_a, real_b, real_c))
        alpha = rs.utils.angle_between(real_b, real_c)
        beta  = rs.utils.angle_between(real_a, real_c)
        gamma = rs.utils.angle_between(real_a, real_b)
        for x in (90., 120.):
            if np.isclose(alpha, x):
                alpha = np.round(alpha)
            if np.isclose(beta, x):
                alpha = np.round(beta)
            if np.isclose(gamma, x):
                alpha = np.round(gamma)
        return [a, b, c, alpha, beta, gamma]

    @staticmethod
    def get_average_cell(expt_file : str) -> gemmi.UnitCell:
        import json
        js = json.load(open(expt_file))
        crystals = js['crystal']
        cells =  []
        for i,exp in js['experiment']:
            cid = exp['crystal']
            crystal = js['crystal'][cid]
            cells.append(
                StillsDataset.real_space_axes_to_cell_params(
                    crystal['real_space_a'],
                    crystal['real_space_b'],
                    crystal['real_space_c'],
                )
            )
        cell = np.array(cells).mean(0)
        return cell

    @staticmethod
    def get_space_group(expt_file : str, check_consistent=False) -> gemmi.SpaceGroup:
        import json
        js = json.load(open(expt_file))

        crystals = js['crystal']
        cells =  []
        sg = None
        for i,exp in js['experiment']:
            cid = exp['crystal']
            crystal = js['crystal'][cid]
            hall = crystal['space_group_hall_symbol']
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
        ds = rs.io.read_dials_stills(self.refl_file)
        js = json.load(open(self.expt_file))

        #Compute wavelength sans experiment list
        ds['wavelength'] = np.reciprocal(np.linalg.norm(ds[['s1.0', 's1.1', 's1.2']], axis=-1))
        ds['wavelength'] = ds.groupby('BATCH')['wavelength'].transform('mean')

        #Remove systematic absences 
        present = ~self.spacegroup.operations().systematic_absences(table['miller_index'])
        table = table.select(flex.bool(present))

        #Trim to resolution range
        #Update resolution for average cell
        table['d'] = flex.double(self.cell.calculate_d_array(table['miller_index']))
        table = table.select(table['d'] >= self.dmin)

        iobs  = torch.tensor(table['intensity.sum.value'], dtype=torch.float32)[:,None]
        sigiobs  = torch.tensor(np.sqrt(table['intensity.sum.variance']), dtype=torch.float32)[:,None]
        hkl = torch.tensor(table['miller_index'], dtype=torch.int32)
        batch = torch.tensor(table['id'], dtype=torch.int32) + self.batch_offsets[expt_file]

        metadata = torch.concat((
            torch.tensor(table['d'], dtype=torch.float32)[:,None]**-2.,
            torch.tensor(table['delpsical.rad'], dtype=torch.float32)[:,None],
            torch.tensor(table['s1'], dtype=torch.float32),
            torch.tensor(table['xyzobs.mm.value'], dtype=torch.float32)[:,:2],
            torch.tensor(table['xyzcal.mm'], dtype=torch.float32)[:,:2],
        ), axis=-1)
        d = torch.tensor(table['d'], dtype=torch.float32)[:,None]
        wavelength = torch.tensor(table['wavelength'], dtype=torch.float32)[:,None]
        rasu_id = self.rasu_id * torch.ones((self.max_nrefln_per_image,1), dtype=batch.dtype)

        self._tensor_data = {
            "image_id" : batch,
            "hkl_in" : hkl,
            "resolution" : d,
            "wavelength" : wavelength,
            "metadata" : metadata,
            "iobs" : iobs, 
            "sigiobs" : sigiobs,
        }

    def __getitem__(self, idx):
        if self._tensor_data is None:
            self._load()

        if idx in self._image_indices:
            idx = self._image_indices[idx]
        else:
            idx = self._tensor_data['image_id'] == idx

        return {k:v[idx] for k,v in self._refls_tensors.items()}

    def __len__(self):
        return len(self.elist)

