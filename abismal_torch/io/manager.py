from typing import List, Optional, Union

import lightning as L
import torch
from torch.utils.data import ConcatDataset, DataLoader, random_split

from abismal_torch.io.mtz import MTZDataset

        
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

class MTZDataModule(L.LightningDataModule):
    def __init__(
        self,
        mtz_files: Union[str, List[str]],
        batch_size: Optional[int] = 1,
        dmin: Optional[float] = None,
        wavelength: Optional[float] = None,
        test_fraction: Optional[float] = 0.05,
        num_workers: Optional[int] = 0,
        rasu_ids: Optional[List[int]] = None,
    ):
        """
        Load MTZ files using LightningDataModule.

        Args:
            mtz_files (str or list[str]): a path or a list of paths to the MTZ files.
            batch_size (int, optional): The batch size for the data loader (number of images per batch).
            dmin (float, optional): The minimum resolution for the data loader.
            wavelength (float, optional): The wavelength for the data loader.
            test_fraction (float, optional): The fraction of the data to use for testing.
            num_workers (int, optional): The number of workers for Pytorch DataLoader.
            rasu_ids (List[int], optional): List of RASU ids corresponding to each MTZ file.
        """
        super().__init__()
        if isinstance(mtz_files, str):
            mtz_files = [mtz_files]

        datasets = []
        if rasu_ids is None:
            rasu_ids = list(range(len(mtz_files)))
        self._rasu_ids = rasu_ids
        for mtz_file, rasu_id in zip(mtz_files, rasu_ids):
            dataset = MTZDataset(
                mtz_file, dmin=dmin, wavelength=wavelength, rasu_id=rasu_id
            )
            datasets.append(dataset)
        self.dataset = ConcatDataset(datasets)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.test_fraction = test_fraction

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
            self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, 
            collate_fn=collate_fn
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers,
            collate_fn=collate_fn
        )

#    def transfer_batch_to_device(self, batch, device, dataloader_idx):
#        return {
#            key: (value.to(device) if isinstance(value, torch.Tensor) else value)
#            for key, value in batch.items()
#        }
