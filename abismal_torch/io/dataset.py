from typing import Optional,List,Sequence,Union
from torch.utils.data import Dataset
from reciprocalspaceship.decorators import cellify,spacegroupify


class AbismalDataset(Dataset):
    @cellify
    @spacegroupify
    def __init__(
        self,
        cell: Optional[List[float]] = None,
        spacegroup: Optional[str] = None,
        wavelength: Optional[float] = 1.,
        rasu_id: Optional[int] = 0,
        dmin: Optional[float] = None,
    ):
        """
        Base class for all AbismalDatasets. This class should not be used directly. 
        Sublclass this file to add new data types to abismal. 

        Apart from following the requirements of `torch.utils.data.Dataset`, subclasses must
        adhere to the following contract. 
            - Accept cell, spacegroup, wavelength, rasu_id, and dmin kwargs
            - If cell or spacegroup are None, populate them in the subclasses __init__
            - If self.cell or self.spacegroup are changed, obey the new values when filtering by resolution
            - implement _can_handle and _load_tensor_data methods
            - optionally overload __len__ with a lazy version. 
            - overload the classmethod `from_sequence` if more than one input file is required. for instance in the case of dials .expt and .refl file pairs.
        """
        self.cell = cell
        self.spacegroup = spacegroup
        self.wavelength = wavelength
        self.rasu_id = rasu_id
        self.dmin = dmin
        self.reset()

    @classmethod
    def from_sequence(cls, input_files, *args, **kwargs):
        return [cls(f, *args, **kwargs) for f in input_files]

    @staticmethod
    def _can_handle(input_files: Sequence) -> bool:
        raise NotImplementedError("Can this handler parse input_files?")

    @classmethod
    def can_handle(cls, input_files: Union[Sequence, str]) -> bool:
        if isinstance(files, str):
            files = [files]
        return cls._can_handle(files)

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

