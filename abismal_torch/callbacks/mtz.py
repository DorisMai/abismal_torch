import os
from typing import Optional

import lightning.pytorch as L


class MTZSaver(L.Callback):
    def __init__(
        self,
        out_dir: str,
        only_observed: Optional[bool] = True,
        save_every_n_epoch: Optional[int] = 1,
    ):
        """
        Callback to save model outputs as MTZ files.

        Args:
            out_dir (str): Directory to save MTZ files
            only_observed (bool, optional): Whether to save only observed reflections.
            save_every_n_epoch (int, optional): Save MTZ file every n epochs.
        """
        super().__init__()
        self.only_observed = only_observed
        self.out_dir = out_dir
        os.makedirs(out_dir, exist_ok=True)
        self.save_every_n_epoch = save_every_n_epoch

    def on_train_end(self, trainer, pl_module) -> None:
        """Save final MTZ file at end of training"""
        self._save_mtz(trainer, pl_module)

    def on_train_epoch_end(self, trainer, pl_module) -> None:
        """Save intermediate MTZ file at end of each epoch"""
        if trainer.current_epoch % self.save_every_n_epoch == 0:
            self._save_mtz(trainer, pl_module, is_intermediate=True)

    def _save_mtz(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        is_intermediate: bool = False,
    ) -> None:
        """
        Save merged outputs as MTZ files for each RASU. File names are set to be out_<rasu_id>.mtz.

        Args:
            is_intermediate (bool, optional): Whether the outputs are intermediate. If yes, mtz file
                file name is appended with the step number.
        """
        posterior = pl_module.merging_model.surrogate_posterior
        for rasu_id, dataset in enumerate(
            posterior.to_dataset(only_observed=self.only_observed)
        ):
            if is_intermediate:
                dataset.write_mtz(
                    os.path.join(
                        self.out_dir,
                        f"out_{rasu_id}_epoch{trainer.current_epoch:02d}.mtz",
                    )
                )
            else:
                dataset.write_mtz(os.path.join(self.out_dir, f"out_{rasu_id}.mtz"))
