from typing import List, Optional, Sequence

import reciprocalspaceship as rs
from reciprocalspaceship.decorators import spacegroupify,cellify
import torch
from torch.utils.data import Dataset,IterableDataset
import gemmi
import numpy as np
from abismal_torch.io.dataset import AbismalDataset

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
            self.cell = StillsDataset.get_average_cell([self.expt_file])
        if self.spacegroup is None:
            self.spacegroup = StillsDataset.get_space_group([self.expt_file])

    @staticmethod
    def _can_handle(input_files):
        for f in input_files:
            if f.split('.')[-1] not in ("expt", "refl", "pickle"):
                return False
        return True

    @property
    def refl_file(self):
        return self._refl_file

    @refl_file.setter
    def refl_file(self, refl_file):
        if refl_file != self._refl_file:
            self._refl_file = refl_file
            self._tensor_data = None
            self._image_indices = {}

    @property
    def expt_file(self, expt_file):
        return self._expt_file

    @expt_file.setter
    def expt_file(self, expt_file):
        from dxtbx.model.experiment_list import ExperimentListFactory
        self.elist = ExperimentListFactory.from_json_file(expt_file, check_format=False)
        self._expt_file = expt_file
        if self._cell is None:
            self.get_average_cell(elist)
            self.get_space_group(elist)

    @staticmethod
    def get_average_cell(elist) -> gemmi.UnitCell:
        crystals = elist.crystals()
        cell = np.array([c.get_unit_cell().parameters() for c in crystals]).mean(0)
        cell = gemmi.UnitCell(*cell)
        return cell

    @staticmethod
    def get_space_group(elist, check_consistent=False) -> gemmi.SpaceGroup:
        hm = elist.crystals()[0].get_space_group().type().universal_hermann_mauguin_symbol()
        sg = gemmi.SpaceGroup(hm)
        if not check_consistent:
            return sg
        for cid,c in enumerate(elist.crystals()):
            _hm = c.get_space_group().type().universal_hermann_mauguin_symbol()
            if not _hm == hm:
                raise ValueError(f"Crystal {cid} has Universal Hermann Mauguin symbol {_hm} but {hm} was expected")
        return sg

    def _load(self):
        from dials.array_family import flex
        table = self._refls
        elist = self.elist

        batch = flex.size_t(np.array(table['id']))

        table.compute_d(elist)
        table["A_matrix"] = flex.mat3_double( [C.get_A() for C in elist.crystals()] ).select(batch)
        table["s0_vec"] = flex.vec3_double( [e.beam.get_s0() for e in elist] ).select(batch)
        table["wavelength"] = flex.double( [e.beam.get_wavelength() for e in elist] ).select(batch)

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


if __name__=='__main__':
    ds = StillsDataset(
            '/Users/kmdalton/xtal/abismal_examples/cxidb_81/reflection_data/01.json',
            '/Users/kmdalton/xtal/abismal_examples/cxidb_81/reflection_data/01.pickle',
    )

    from IPython import embed
    embed(colors='linux')

