import os
from typing import Optional

import lightning.pytorch as L

class StandardizationFreezer(L.Callback):
    def __init__(
        self,
        max_tracking_batches: Optional[int] = None,
        max_tracking_epochs: Optional[int] = None,
    ):
        """
        Callback to disable tracking running stats in standardization layers of the merging model
        after a specified number of batches or epochs.

        Args:
            max_tracking_batches (int, optional): Disable tracking after this many batches.
                If None, batch-based tracking is not used. Default: None
            max_tracking_epochs (int, optional): Disable tracking after this many epochs.
                If None, epoch-based tracking is not used. Default: None
        
        Note:
            If both are specified, tracking will be disabled when either threshold is reached.
        """
        super().__init__()
        self.max_tracking_batches = max_tracking_batches
        self.max_tracking_epochs = max_tracking_epochs
        if self.max_tracking_batches is not None and self.max_tracking_epochs is not None:
            raise ValueError("Only one of max_tracking_batches or max_tracking_epochs can be specified")
        if self.max_tracking_batches is None and self.max_tracking_epochs is None:
            raise ValueError("One of max_tracking_batches or max_tracking_epochs must be specified")

    def _disable_tracking(self, pl_module: L.LightningModule) -> None:
        """Disable tracking in both standardization layers."""
        pl_module.merging_model.standardize_intensity.track_running_stats = False
        pl_module.merging_model.standardize_metadata.track_running_stats = False

    def on_train_batch_end(
        self, 
        trainer: L.Trainer, 
        pl_module: L.LightningModule,
        outputs,
        batch,
        batch_idx: int,
    ) -> None:
        """Check if we should disable tracking based on batch count."""
        if self.max_tracking_batches is not None:
            # global_step is 0-indexed, so we check >= max_tracking_batches
            # This means after max_tracking_batches batches, tracking stops
            if trainer.global_step >= self.max_tracking_batches:
                self._disable_tracking(pl_module)

    def on_train_epoch_end(
        self, 
        trainer: L.Trainer, 
        pl_module: L.LightningModule
    ) -> None:
        """Check if we should disable tracking based on epoch count."""
        if self.max_tracking_epochs is not None:
            # current_epoch is 0-indexed, so after max_tracking_epochs epochs,
            # current_epoch will be >= max_tracking_epochs
            if trainer.current_epoch >= self.max_tracking_epochs:
                self._disable_tracking(pl_module)

    

    