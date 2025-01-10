from typing import Optional

import lightning as L
import torch
from torch.utils.data import DataLoader, random_split

from abismal_torch.io.mtz import MTZDataset


class MTZDataModule(L.LightningDataModule):
    def __init__(
        self,
        mtz_file: str,
        batch_size: Optional[int] = 1,
        dmin: Optional[float] = None,
        wavelength: Optional[float] = None,
        test_fraction: Optional[float] = 0.05,
        num_workers: Optional[int] = 1,
    ):
        super().__init__()
        self.dataset = MTZDataset(mtz_file, dmin=dmin, wavelength=wavelength)
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
