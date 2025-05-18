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
        dmin: float,
        batch_size: Optional[int] = 1,
        wavelength: Optional[float] = None,
        validation_fraction: Optional[float] = 0.05,
        test_fraction: Optional[float] = 0.05,
        num_workers: Optional[int] = 0,
        rasu_ids: Optional[List[int]] = None,
        anomalous: Optional[bool] = False,
        cell: Optional[List[float]] = None,
        spacegroup: Optional[str] = None,
        pin_memory: Optional[bool] = False,
        persistent_workers: Optional[bool] = False,
    ):
        """
        Load MTZ files using LightningDataModule.

        Args:
            mtz_files (str or list[str]): a path or a list of paths to the MTZ files.
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
        """
        super().__init__()
        self.dmin = dmin
        self.anomalous = anomalous
        if isinstance(mtz_files, str):
            mtz_files = [mtz_files]

        self.num_asus = len(mtz_files)
        datasets = []
        # if rasu_ids is None:
        #     rasu_ids = list(range(self.num_asus))
        if rasu_ids is not None:
            mtz_files = [x for _, x in sorted(zip(rasu_ids, mtz_files))]
        rasu_ids = list(range(self.num_asus))
        for mtz_file, rasu_id in zip(mtz_files, rasu_ids):
            dataset = MTZDataset(
                mtz_file,
                dmin=dmin,
                wavelength=wavelength,
                rasu_id=rasu_id,
                cell=cell,
                spacegroup=spacegroup,
            )
            datasets.append(dataset)
        self.cell = [dataset.cell for dataset in datasets]
        self.spacegroup = [dataset.spacegroup for dataset in datasets]
        self.dataset = ConcatDataset(datasets)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.validation_fraction = validation_fraction
        self.test_fraction = test_fraction
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers

    def setup(self, stage: Optional[str] = None):
        # Random split based on images, not reflections
        total_len = len(self.dataset)
        val_size = int(total_len * self.validation_fraction)
        test_size = int(total_len * self.test_fraction)
        train_size = total_len - val_size - test_size
        self.train_dataset, self.val_dataset, self.test_dataset = random_split(
            self.dataset,
            [train_size, val_size, test_size],
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

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
        )
