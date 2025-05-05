from typing import List, Optional, Sequence

import reciprocalspaceship as rs
import torch
from torch.utils.data import Dataset,IterableDataset
import gemmi
import numpy as np


class StillsDataset(IterableDataset):
    def __init__(
        self,
        expt_files: Sequence[str],
        refl_files: Sequence[str],
        cell: Optional[List[float]] = None,
        spacegroup: Optional[str] = None,
        wavelength: Optional[float] = None,
        rasu_id: Optional[int] = 0,
        dmin: Optional[float] = 0.,
        max_nrefln_per_image: Optional[int] = 512,
    ):
        """
        Custom Pytorch Dataset Class for DIALS Data

        Args:
            expt_files (Sequence[str]): An interable of experiment file names
            refl_files (Sequence[str]): An interable of reflection file names
            cell (list[float], optional): a list of cell parameters. If provided, overrides the cell parameters in the MTZ file.
            spacegroup (str, optional): a spacegroup symbol. If provided, overrides the spacegroup in the MTZ file.
            wavelength (float, optional): a wavelength. By default this will be read from the .expt files. 
            rasu_id (int, optional): a rasu id. Defaults to 0.
            dmin (float, optional): Highest resolution to include.
            max_nrefln_per_image (int, optional): Maximum number of reflections per image to include in a training batch. This is
                used to avoid loading too many reflections into memory at once. Defaults to 2048.
        """
        super().__init__()
        self.cell = cell
        self.spacegroup = spacegroup
        self.wavelength = wavelength
        self.rasu_id = rasu_id
        self.dmin = dmin
        self.expt_files = expt_files
        self.refl_files = refl_files
        if len(expt_files) != len(refl_files):
            raise ValueError("len(expt_files) is not the same as len(refl_files)")

        # Try to be smart and use the inputs to infer the symmetry 
        if self.cell is None:
            self.cell = StillsDataset.get_average_cell(expt_files)
        if self.spacegroup is None:
            self.spacegroup = StillsDataset.get_space_group(expt_files)
        self.max_nrefln_per_image = max_nrefln_per_image

        from dxtbx.model.experiment_list import ExperimentListFactory
        self.batch_offsets = {}
        count = 0
        for efile in expt_files:
            self.batch_offsets[efile] = count
            elist = ExperimentListFactory.from_json_file(efile, check_format=False)
            count = count + len(elist)
        self.count_max = count

    @staticmethod
    def get_average_cell(expt_files: Sequence[str]) -> gemmi.UnitCell:
        from dxtbx.model.experiment_list import ExperimentListFactory
        cell = np.zeros(6)
        l = 0
        from tqdm import tqdm
        print("Determining unit cell ...")
        for efile in tqdm(expt_files):
            elist = ExperimentListFactory.from_json_file(efile, check_format=False)
            crystals = elist.crystals()
            cell += np.array([c.get_unit_cell().parameters() for c in crystals]).sum(0)
            l += len(crystals)
        cell = cell/l
        cell = gemmi.UnitCell(*cell)
        print(f"Average cell: {cell}")
        return cell
        from dxtbx.model.experiment_list import ExperimentListFactory

    @staticmethod
    def get_space_group(expt_files: Sequence[str], check_consistent=False) -> gemmi.SpaceGroup:
        from dxtbx.model.experiment_list import ExperimentListFactory
        hm = None
        for efile in expt_files:
            elist = ExperimentListFactory.from_json_file(efile, check_format=False)
            hm = elist.crystals()[0].get_space_group().type().universal_hermann_mauguin_symbol()
            if not check_consistent:
                sg = gemmi.SpaceGroup(hm)
                return sg
            for cid,c in enumerate(elist.crystals()):
                _hm = c.get_space_group().type().universal_hermann_mauguin_symbol()
                if not _hm == hm:
                    raise ValueError(f"Crystal {cid} from {efile} has Universal Hermann Mauguin symbol {_hm} but {hm} was expected")
        hm = gemmi.SpaceGroup(hm)
        return sg

    def iter_images(self, expt_file: Sequence[str], refl_file: Sequence[str]):
        from dials.array_family import flex
        from dxtbx.model.experiment_list import ExperimentListFactory 
        table = flex.reflection_table().from_file(refl_file)
        elist = ExperimentListFactory.from_json_file(expt_file, check_format=False)

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

        for m in torch.unique(batch):
            idx = batch == m
            image = {
                "image_id" : batch[idx,None],
                "hkl_in" : hkl[idx],
                "resolution" : d[idx],
                "wavelength" : wavelength[idx],
                "metadata" : metadata[idx],
                "iobs" : iobs[idx], 
                "sigiobs" : sigiobs[idx],
            }

            #Resample uniform shape
            idx = torch.randint(idx.sum(), size=(self.max_nrefln_per_image,))
            image = {k:v[idx] for k,v in image.items()}
            image["rasu_id"] = rasu_id
            yield image

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        num_workers = 1
        worker_id = 0
        if worker_info is not None:
            num_workers = worker_info.num_workers
            worker_id = worker_info.id
        file_ids = np.arange(len(self.expt_files))
        file_ids = np.array_split(file_ids, num_workers)[worker_id]
        for file_id in file_ids:
            expt_file = self.expt_files[file_id]
            refl_file = self.refl_files[file_id]
            for image in self.iter_images(expt_file, refl_file):
                yield image

