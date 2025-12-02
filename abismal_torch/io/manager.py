from cgitb import handler
from typing import Mapping, Optional, Sequence

import gemmi
import lightning as L
import torch
from torch.utils.data import DataLoader, random_split

from abismal_torch.io.dataset import AbismalConcatDataset
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


class AbismalDataModule(L.LightningDataModule):
    handlers = {
        MTZDataset.__HANDLER_TYPE__: MTZDataset,
        StillsDataset.__HANDLER_TYPE__: StillsDataset,
    }

    def __init__(
        self,
        rasu_configs: Mapping | Sequence[Mapping],
        batch_size: Optional[int] = 1,
        validation_fraction: Optional[float] = 0.05,
        test_fraction: Optional[float] = 0.05,
        num_workers: Optional[int] = 1,
        pin_memory: Optional[bool] = False,
        persistent_workers: Optional[bool] = False,
        handler_kwargs: Optional[Mapping] = {},
    ):
        """
        Load files using LightningDataModule. This module supports various configurations for the
        symmetry. The unit cells and/or spacegroups can be fully specified or inferred from the data.
        If cell or spacegroup are not specificed, they will be inferred on a per-rasu basis.

        Args:
            rasu_configs (dict): A dictionary or list of dictionaries of configuration for each rasu.
                Configuration keys are:
                    - rasu_id (int): The RASU id corresponding to the input files, must be 0 indexed.
                    - input_files (str or Sequence[str]): a path or a list of paths to the reflection files.
                    - dmin (float, optional): The highest resolution limit. Defaults to 0.
                    - wavelength (float, optional): The wavelength for the data loader. If not provided, will
                        be inferred from the data or default to 1.0.
                    - anomalous (bool, optional): Whether the data is anomalous. Defaults to False.
                    - cell (gemmi.UnitCell or List[float]): The unit cell constants. If not provided, will be
                        inferred from the data as weighted average of images.
                    - spacegroup (gemmi.SpaceGroup or str]): The spacegroups. If not provided, will be inferred
                        from the data.
            batch_size (int, optional): The batch size for the data loader (number of images per batch).
            test_fraction (float, optional): The fraction of the data to use for testing.
            num_workers (int, optional): The number of workers for Pytorch DataLoader.
            pin_memory (bool, optional): Whether to pin memory.
            persistent_workers (bool, optional): Whether workers are persistent.
            **handler_kwargs (optional): Additional keyword arguments to pass to the file handler
        """
        super().__init__()
        if isinstance(rasu_configs, dict):
            rasu_configs = [rasu_configs]
        self.anomalouss = {}
        self.datasets = []
        _used_handler_types = set()
        for rasu_config in rasu_configs:
            # get the anomalous flag for rasu
            rasu_id = rasu_config.get("rasu_id", 0)
            anomalous = rasu_config.pop("anomalous", False)
            if rasu_id in self.anomalouss and self.anomalouss[rasu_id] != anomalous:
                raise ValueError(f"Inconsistent anomalous flags for rasu_id {rasu_id}")
            self.anomalouss[rasu_id] = anomalous
            # parse the input files
            input_files = rasu_config.pop("input_files")
            if isinstance(input_files, str):
                input_files = [input_files]
            handler_type_2_input_files = {}
            for input_file in input_files:
                handler_type = self.determine_handler_type(input_file)
                if handler_type not in handler_type_2_input_files:
                    handler_type_2_input_files[handler_type] = []
                handler_type_2_input_files[handler_type].append(input_file)
            # construct the datasets for this rasu
            _used_handler_types.update(handler_type_2_input_files.keys())
            for k, v in handler_type_2_input_files.items():
                self.datasets.extend(
                    AbismalDataModule.handlers[k].from_sequence(
                        v, **rasu_config, **handler_kwargs
                    )
                )

        if len(self.anomalouss) != max(self.anomalouss.keys()) + 1:
            raise ValueError("rasu_ids must form contiguous sequence starting from 0.")

        # handle metadata keys for mixing handler types
        _used_handler_types = sorted(_used_handler_types)
        _used_handler_metadata_lengths = []
        if len(_used_handler_types) > 1:
            _used_handler_2_metadata_length = {
                handler_type: len(
                    AbismalDataModule.handlers[handler_type].__DEFAULT_METADATA_KEYS__
                )
                for handler_type in _used_handler_types
            }
            for k, v in handler_kwargs.items():
                if k.endswith("_metadata_keys"):
                    handler_type = k.split("_metadata_keys")[0]
                    _used_handler_2_metadata_length[handler_type] = len(v)
            _used_handler_metadata_lengths = [
                _used_handler_2_metadata_length[handler_type]
                for handler_type in _used_handler_types
            ]

        self.dataset = AbismalConcatDataset(
            self.datasets, _used_handler_types, _used_handler_metadata_lengths
        )
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.validation_fraction = validation_fraction
        self.test_fraction = test_fraction
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers

    @property
    def dmins(self):
        return self.dataset.dmins

    @property
    def cells(self):
        return self.dataset.cells

    @property
    def spacegroups(self):
        return self.dataset.spacegroups

    @staticmethod
    def determine_handler_type(input_files: str) -> str:
        """
        Determine the handler types for the input files. Supports a mixture of file types.
        """
        for k, v in AbismalDataModule.handlers.items():
            if v.can_handle([input_files]):
                return k
        raise ValueError(
            f"Cannot determine the parser to handle the file: {input_files}"
        )

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
            shuffle=True,
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


#    def transfer_batch_to_device(self, batch, device, dataloader_idx):
#        return {
#            key: (value.to(device) if isinstance(value, torch.Tensor) else value)
#            for key, value in batch.items()
#        }
