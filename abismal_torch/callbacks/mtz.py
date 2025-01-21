import os
from typing import Optional

import lightning.pytorch as L


class MTZSaver(L.Callback):
    def __init__(self, out_dir: str, only_observed: Optional[bool] = True):
        """
        Callback to save model outputs as MTZ files.

        Args:
            out_dir (str): Directory to save MTZ files
            only_observed (bool, optional): Whether to save only observed reflections.
        """
        super().__init__()
        self.only_observed = only_observed
        self.out_dir = out_dir
        os.makedirs(out_dir, exist_ok=True)
        # self.save_every_n_epochs = save_every_n_epochs

    # def on_validation_epoch_end(
    #     self, trainer: L.Trainer, pl_module: L.LightningModule
    # ) -> None:
    #     """Save MTZ file after validation if save_every_n_epochs is set"""
    #     if self.save_every_n_epochs is None:
    #         return

    #     if (trainer.current_epoch + 1) % self.save_every_n_epochs == 0:
    #         self._save_mtz(trainer, pl_module)

    def on_train_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        """Save final MTZ file at end of training"""
        self._save_mtz(trainer, pl_module)

    def _save_mtz(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        """
        Save current model outputs as MTZ file
        """
        posterior = pl_module.merging_model.surrogate_posterior
        for rasu_id, dataset in enumerate(
            posterior.to_dataset(only_observed=self.only_observed)
        ):
            dataset.write_mtz(os.path.join(self.out_dir, f"out_{rasu_id}.mtz"))
