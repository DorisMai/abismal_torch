from typing import List, Optional, Union, Sequence

import lightning as L
import torch
from torch.utils.data import ConcatDataset, DataLoader, random_split
from reciprocalspaceship.decorators import cellify,spacegroupify

from abismal_torch.io.mtz import MTZDataset
from abismal_torch.io.stills import StillsDataset
        
def collate_fn(batch):
    """
    Custom collate function to handle batches of image reflections.
    Make sure this is a top-level function, not a method of a class, otherwise
    it will not be pickled when using multiprocessing.

    Args:
        batch: List of dictionaries, where each dict contains tensor arrays
              for a single image's reflections

    Returns:
        Dictionary with batched tensors
    """
    # If batch_size=1, just return the single image's data
    if len(batch) == 1:
        return batch[0]

    result = {}
    for key in batch[0].keys():
        result[key] = torch.cat([item[key] for item in batch])

    return result

class IsomorphousDataModule(L.LightningDataModule):
    handlers = {
        'mtz' : MTZDataset,
        'dials' : StillsDataset,
    }
    @cellify
    @spacegroupify
    def __init__(
        self,
        input_files: Union[str, Sequence[str]],
        dmin : float,
        batch_size: Optional[int] = 1,
        wavelength: Optional[float] = None,
        test_fraction: Optional[float] = 0.05,
        num_workers: Optional[int] = 0,
        rasu_ids: Optional[Sequence[int]] = None,
        anomalous: Optional[bool] = False,
        cell: Optional[Union[gemmi.UnitCell, Sequence[float]]] = None,
        spacegroup: Optional[Union[gemmi.SpaceGroup, str, int]] = None,
        pin_memory: Optional[bool] = False,
        persistent_workers: Optional[bool] = False,
        **handler_kwargs: Optional,
    ):
        """
        Load MTZ files using LightningDataModule.

        Args:
            input_files (str or Sequence[str]): a path or a list of paths to the reflection files.
            dmin (float): The highest resolution limit.
            batch_size (int, optional): The batch size for the data loader (number of images per batch).
            wavelength (float, optional): The wavelength for the data loader.
            test_fraction (float, optional): The fraction of the data to use for testing.
            num_workers (int, optional): The number of workers for Pytorch DataLoader.
            rasu_ids (List[int], optional): List of RASU ids corresponding to each MTZ file. If supplied, make sure they are unique.
                They will be renumberd from 0 to num_asus-1 according to the ascending order of the list if supplied, or according
                to the order of the MTZ files if not supplied.
            anomalous (bool, optional): Whether the data is anomalous.
            cell (list[float], optional): a list of cell parameters. If provided, overrides the cell parameters in the MTZ file.
            spacegroup (str, optional): a spacegroup symbol. If provided, overrides the spacegroup in the MTZ file.
            pin_memory (bool, optional): Whether to pin memory.
            persistent_workers (bool, optional): Whether workers are persistent.
            **handler_keargs (optional): Additional keyword arguments to pass to the file handler
        """
        super().__init__()
        self.dmin = dmin
        self.anomalous = anomalous

        handler_type = self.determine_handler_type(input_files)
        handler = self.handlers[handler_type]
        datasets = handler.from_sequence(
            input_files, dmin=dmin, wavelength=wavelength, rasu_id=rasu_id, 
            cell=cell, spacegroup=spacegroup
        )

        self.cell = np.zeros(6)
        count = 0
        for ds in datasets:
            n = len(ds)
            cell = np.array(ds.cell.parameters())
            count += n
        self.cell = self.cell / count
        self.cell = gemmi.UnitCell(*cell)

        self.spacegroup = datasets[0].spacegroup
        for ds in datasets:
            ds.spacegroup = spacegroup
            ds.cell = self.cell
        self.dataset = ConcatDataset(datasets)

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.test_fraction = test_fraction
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers

    def determine_handler_type(input_files: Union[str, Sequence[str]]) -> str:
        for k,v in handler:
            if v.can_handle(input_files):
                return k

    def setup(self, stage: Optional[str] = None):
        # Random split based on images, not reflections
        total_len = len(self.dataset)
        val_size = int(total_len * self.test_fraction)
        train_size = total_len - val_size
        self.train_dataset, self.val_dataset = random_split(
            self.dataset,
            [train_size, val_size],
            generator=torch.Generator(),
        )

    def train_dataloader(self):
        # Each item is already a batch of reflections from a single image
        # batch_size=1 means process one image at a time (with all its reflections)
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
        )


#    def transfer_batch_to_device(self, batch, device, dataloader_idx):
#        return {
#            key: (value.to(device) if isinstance(value, torch.Tensor) else value)
#            for key, value in batch.items()
#        }
