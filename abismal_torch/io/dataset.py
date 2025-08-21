from typing import Optional,List,Sequence,Union,Iterable
import torch
from torch.utils.data import Dataset,ConcatDataset
from reciprocalspaceship.decorators import cellify,spacegroupify
import numpy as np
import gemmi
import bisect


class AbismalDataset(Dataset):
    __DEFAULT_METADATA_KEYS__ = []
    __HANDLER_TYPE__ = None

    def __init__(
        self,
        cell: Optional[List[float]] = None,
        spacegroup: Optional[str] = None,
        wavelength: Optional[float] = 1.,
        rasu_id: Optional[int] = 0,
        dmin: Optional[float] = None,
        **kwargs,
    ):
        """
        Base class for all AbismalDatasets. This class should not be used directly. 
        Subclass this file to add new data types to abismal. 

        Apart from following the requirements of `torch.utils.data.Dataset`, subclasses must
        adhere to the following contract. 
            - Accept cell, spacegroup, wavelength, rasu_id, and dmin kwargs
            - If cell or spacegroup are None, populate them in the subclasses __init__
            - If self.cell or self.spacegroup are changed, obey the new values when filtering by resolution
            - implement _can_handle and _load_tensor_data methods
            - optionally overload __len__ with a lazy version. 
            - overload the classmethod `from_sequence` if more than one input file is required. for instance in the case of dials .expt and .refl file pairs.
            - overload the __DEFAULT_METADATA_KEYS__ class attribute if the dataset has a different set of metadata keys.

        kwargs allow subclass to accept additional arguments while ignore irrelevant arguments from the manager.
        """
        self.cell = cell
        self.spacegroup = spacegroup
        self.wavelength = wavelength
        self.rasu_id = rasu_id
        self.dmin = dmin
        self.reset()

    @classmethod
    def from_sequence(
            cls, 
            input_files: Iterable[str], 
            dmin : List[float] | float = 0., 
            wavelength : List[float] | float = 1.0,
            rasu_id : List[int] | int = 0,
            cell : List[gemmi.UnitCell] | gemmi.UnitCell = None,
            spacegroup : List[gemmi.SpaceGroup] | gemmi.SpaceGroup = None,
            **kwargs,
        ) -> List:
        kwargs.update({
            'dmin' : dmin,
            'wavelength' : wavelength,
            'rasu_id' : rasu_id,
            'cell' : cell,
            'spacegroup' : spacegroup,
        })
        length = len(input_files)
        for k,v in kwargs.items():
            if (not isinstance(v, list)) or (not isinstance(v, tuple)):
                kwargs[k] = [v] * length
        result = []
        for i,f in enumerate(input_files):
            result.append(cls(f, **{k : v[i] for k,v in kwargs.items()}))
        return result

    @staticmethod
    def _can_handle(input_files: Sequence) -> bool:
        raise NotImplementedError("Can this handler parse input_files?")

    @classmethod
    def can_handle(cls, input_files: Union[Sequence, str]) -> bool:
        if isinstance(input_files, str):
            input_files = [input_files]
        return cls._can_handle(input_files)

    @property
    def rasu_id(self):
        return self._rasu_id

    @rasu_id.setter
    def rasu_id(self, rasu_id):
        self._rasu_id = rasu_id
        self.reset()

    @property
    def wavelength(self):
        return self._wavelength

    @wavelength.setter
    def wavelength(self, wavelength):
        self._wavelength = wavelength
        self.reset()

    @property
    def cell(self):
        return self._cell

    @cell.setter
    @cellify
    def cell(self, cell):
        self._cell = cell
        self.reset()

    @property
    def dmin(self):
        return self._dmin

    @property
    def spacegroup(self):
        return self._spacegroup

    @spacegroup.setter
    @spacegroupify
    def spacegroup(self, spacegroup):
        self._spacegroup = spacegroup
        self.reset()

    @property
    def dmin(self):
        return self._dmin

    @dmin.setter
    def dmin(self, dmin):
        self._dmin = dmin
        self.reset()

    def reset(self):
        self._tensor_data = None
        self._image_data = {}
        self._length = None

    def _load_tensor_data(self):
        """
        Populate the self._tensor_data attribute
        """
        raise NotImplementedError(
            "Subclasses must implement _load_tensor_data to populate self._tensor_data"
        )

    @property
    def tensor_data(self) -> dict:
        if self._tensor_data is None:
            self._load_tensor_data()
        return self._tensor_data

    def __len__(self) -> int:
        """
        Expensive, fallback implementation of __len__. If possible overload this to avoid
        calling _load_tensor_data. Alternatively you can set self._length.
        """
        if self._length is None:
            self._length = self.tensor_data['image_id'].max() + 1
        return self._length

    def __getitem__(self, idx):
        """
        Returns a single image worth of data as a dictionary with the following structure
        
        ```python
        {
            'image_id': torch.tensor[int32] of shape (n_refls,),
            'rasu_id': torch.tensor[int32] of shape (n_refls,),
            'resolution': torch.tensor[float32] of shape (n_refls,),
            'wavelength': torch.tensor[float32] of shape (n_refls,),
            'metadata': torch.tensor[float32] of shape (n_refls, n_metadata),
            'iobs': torch.tensor[float32] of shape (n_refls,),
            'sigiobs': torch.tensor[float32] of shape (n_refls,),
        }
        ```

        Args:
            idx (int): an integer index

        Returns:
            datum (dict): A dictionary with the following keys
        """
        l = len(self)
        if (idx > l -1)  | (idx < -l):
            raise IndexError(f"Index {idx} out of range for AbismalDataset with length {l}")
        if idx in self._image_data:
            return self._image_data[idx]

        mask = self.tensor_data['image_id'] == idx
        self._image_data[idx] = {k:v[mask] for k,v in self._tensor_data.items()}
        mask = ~mask
        self._tensor_data = {k:v[mask] for k,v in self._tensor_data.items()}
        return self._image_data[idx]

class AbismalConcatDataset(ConcatDataset):
    """
    This is a subclass of torch.utils.data.ConcatDataset which is aware of the
    symmetry of AbismalDatasets.
    """
    def __init__(self, datasets: List[Dataset], 
                 handler_types: Optional[List[str]] = [], 
                 handler_metadata_lengths: Optional[List[int]] = []) -> None:
        """
        Args:
            datasets (List[Dataset]): A list of AbismalDatasets, which can be longer
              than the number of asus.
        
        Attributes:
            lengths (dict): A dictionary of rasu_id to number of images.
            cells (dict): A dictionary of rasu_id to cell.
            spacegroups (dict): A dictionary of rasu_id to spacegroup.
            dmins (dict): A dictionary of rasu_id to dmin.
        """
        super().__init__(datasets)

        cells = self.cells
        spacegroups = self.spacegroups
        for ds in self.datasets:
            ds.cell = cells[ds.rasu_id]
            ds.spacegroup = spacegroups[ds.rasu_id]

        self._handler_types = handler_types
        self._handler_metadata_lengths = handler_metadata_lengths
        self._zero_padding = False
        if len(self._handler_types) > 1:
            if len(self._handler_metadata_lengths) != len(self._handler_types):
                raise ValueError("If mixing handler types, handler_types and handler_metadata_lengths must be the same length")
            self._zero_padding = True

        print("for each dataset, the metadata keys are:")
        for ds in self.datasets:
            print(ds.metadata_keys)

    @property
    def lengths(self):
        lengths = {}
        for ds in self.datasets:
            if ds.rasu_id not in lengths:
                lengths[ds.rasu_id] = 0
            lengths[ds.rasu_id] += len(ds)
        return lengths

    @property
    def cells(self):
        cells = {}
        lens = self.lengths
        for ds in self.datasets:
            if ds.rasu_id not in cells:
                cells[ds.rasu_id] = np.zeros(6)
            cells[ds.rasu_id] += len(ds) * np.array(ds.cell.parameters) / lens[ds.rasu_id]
        cells = {k:gemmi.UnitCell(*v) for k,v in cells.items()}
        return cells

    @property
    def spacegroups(self):
        spacegroups = {}
        for ds in self.datasets:
            if ds.rasu_id in spacegroups:
                assert spacegroups[ds.rasu_id] == ds.spacegroup, f"Inconsistent spacegroups for rasu_id {ds.rasu_id}"
            spacegroups[ds.rasu_id] = ds.spacegroup
        return spacegroups

    @property
    def dmins(self):
        dmins = {}
        for ds in self.datasets:
            if ds.rasu_id in dmins:
                assert dmins[ds.rasu_id] == ds.dmin, f"Inconsistent dmins for rasu_id {ds.rasu_id}"
            dmins[ds.rasu_id] = ds.dmin
        return dmins
    
    def __getitem__(self, idx):
        if idx < 0:
            if -idx > len(self):
                raise ValueError(
                    "absolute value of index should not exceed dataset length"
                )
            idx = len(self) + idx
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]

        sample = self.datasets[dataset_idx][sample_idx]
        if self._zero_padding and (sample['metadata'].shape[1] != sum(self._handler_metadata_lengths)):
            handler_idx = self._handler_types.index(self.datasets[dataset_idx].__HANDLER_TYPE__)
            start_idx = sum(self._handler_metadata_lengths[:handler_idx])
            end_idx = start_idx + self._handler_metadata_lengths[handler_idx]
            num_reflections = sample['metadata'].shape[0]
            _padded_metadata = torch.zeros((num_reflections, sum(self._handler_metadata_lengths)), dtype=sample['metadata'].dtype)
            _padded_metadata[:,start_idx:end_idx] = sample['metadata']
            sample['metadata'] = _padded_metadata
        return sample