from typing import List, Optional, Union

import lightning as L
import torch
from torch.utils.data import ConcatDataset, DataLoader, random_split

from abismal_torch.io.mtz import MTZDataset


class MTZDataModule(L.LightningDataModule):
    def __init__(
        self,
        mtz_files: Union[str, List[str]],
        batch_size: Optional[int] = 1,
        dmin: Optional[float] = None,
        wavelength: Optional[float] = None,
        test_fraction: Optional[float] = 0.05,
        num_workers: Optional[int] = 1,
        rasu_ids: Optional[List[int]] = None,
    ):
        """
        Load MTZ files using LightningDataModule.

        Args:
            mtz_files (str or list[str]): a path or a list of paths to the MTZ files.
            batch_size (int, optional): The batch size for the data loader.
            dmin (float, optional): The minimum resolution for the data loader.
            wavelength (float, optional): The wavelength for the data loader.
            test_fraction (float, optional): The fraction of the data to use for testing.
            num_workers (int, optional): The number of workers for Pytorch DataLoader.
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
        self.train_dataset, self.val_dataset = random_split(
            self.dataset,
            [1 - self.test_fraction, self.test_fraction],
            generator=torch.Generator(),
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers
        )

#    def transfer_batch_to_device(self, batch, device, dataloader_idx):
#        return {
#            key: (value.to(device) if isinstance(value, torch.Tensor) else value)
#            for key, value in batch.items()
#        }
