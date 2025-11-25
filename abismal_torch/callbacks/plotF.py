import os
from typing import Optional

import lightning.pytorch as L
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import torch


class PosteriorPlotter(L.Callback):
    def __init__(self, save_every_n_epoch: int = 1, use_wandb: bool = True, scatter_kwargs: Optional[dict] = None):
        """
        Callback to plot F vs SIGF from loc and scale of the posterior distribution.

        Args:
            save_every_epoch (bool, optional): Whether to save the plot every epoch.
            use_wandb (bool, optional): Whether to use wandb to log the plot. If False,
                the plot will be saved to the logger.save_dir. Defaults to True.
        """
        super().__init__()
        self.save_every_n_epoch = save_every_n_epoch
        default_kwaargs = {"alpha": 0.5, "s": 10}
        if scatter_kwargs is not None:
            default_kwaargs.update(scatter_kwargs)
        self.scatter_kwaargs = default_kwaargs
        self.use_wandb = use_wandb

    def on_train_epoch_end(
        self, trainer: L.Trainer, pl_module: L.LightningModule
    ) -> None:
        fig = self._plot_posterior(trainer, pl_module)
        if self.use_wandb:
            import wandb
            if trainer.current_epoch % self.save_every_n_epoch == 0:
                wandb.log(
                    {
                        "posterior": [
                            wandb.Image(fig, caption=f"Epoch {trainer.current_epoch}")
                        ]
                    }
                )
        else:
            fig.savefig(os.path.join(trainer.logger.save_dir, f"posterior_epoch{trainer.current_epoch}.png"))
        plt.close(fig)

    def on_train_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        fig = self._plot_posterior(trainer, pl_module)
        if self.use_wandb:
            import wandb
            wandb.log(
                {"posterior": [wandb.Image(fig, caption="Final posterior")]},
                step=trainer.current_epoch,
            )
        else:
            fig.savefig(os.path.join(trainer.logger.save_dir, f"posterior_final.png"))
        plt.close(fig)

    def _plot_posterior(
        self, trainer: L.Trainer, pl_module: L.LightningModule
    ) -> plt.Figure:
        surrogate_posterior = pl_module.merging_model.surrogate_posterior
        with torch.no_grad():
            loc = surrogate_posterior.distribution.loc.detach().cpu().numpy()
            scale = surrogate_posterior.distribution.scale.detach().cpu().numpy()
            fig, ax = plt.subplots()
            ax.scatter(scale, loc, **self.scatter_kwaargs)
            ax.set_box_aspect(1)
            ax.set_xlabel("scale")
            ax.set_ylabel("loc")
            plt.tight_layout()
            return fig
