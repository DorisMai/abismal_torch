import os
from typing import Optional

import lightning.pytorch as L
import matplotlib.pyplot as plt
import torch
import wandb


class PosteriorPlotter(L.Callback):
    def __init__(self, save_every_epoch: bool = True):
        """
        Callback to plot F vs SIGF from loc and scale of the posterior distribution.

        Args:
            save_every_epoch (bool, optional): Whether to save the plot every epoch.
        """
        super().__init__()
        self.save_every_epoch = save_every_epoch
        self.plot_kwaargs = {"alpha": 0.5, "s": 10}

    def on_train_epoch_end(
        self, trainer: L.Trainer, pl_module: L.LightningModule
    ) -> None:
        if self.save_every_epoch:
            fig = self._plot_posterior(trainer, pl_module)
            # fig.savefig(os.path.join(trainer.logger.save_dir, f"posterior_epoch{trainer.current_epoch}.png"))
            wandb.log(
                {
                    "posterior": [
                        wandb.Image(fig, caption=f"Epoch {trainer.current_epoch}")
                    ]
                }
            )
            plt.close(fig)

    def _plot_posterior(
        self, trainer: L.Trainer, pl_module: L.LightningModule
    ) -> plt.Figure:
        surrogate_posterior = pl_module.merging_model.surrogate_posterior
        with torch.no_grad():
            loc = surrogate_posterior.distribution.loc.detach().cpu().numpy()
            scale = surrogate_posterior.distribution.scale.detach().cpu().numpy()
            fig, ax = plt.subplots()
            ax.scatter(scale, loc, **self.plot_kwaargs)
            ax.set_box_aspect(1)
            ax.set_xlabel("scale")
            ax.set_ylabel("loc")
            plt.tight_layout()
            return fig
