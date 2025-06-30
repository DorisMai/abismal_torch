from typing import Any, Optional

import lightning as L
import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.utilities import grad_norm
from pytorch_lightning.loggers import WandbLogger
from torch.optim import Adam

from abismal_torch.callbacks import MTZSaver, PosteriorPlotter


class AbismalLitModule(L.LightningModule):
    def __init__(
        self, merging_model, optimizer_kwargs: dict, kl_weight: Optional[float] = 1.0
    ):
        """
        Args:
            merging_model (torch.nn.Module): Variational merging model.
            kl_weight (float, optional): KL divergence weight. Defaults to 1.0.
            epsilon (float, optional): Epsilon for numerical stability. Defaults to 1e-6.
        """
        super().__init__()
        self.save_hyperparameters(ignore=["merging_model"])
        self.merging_model = merging_model
        self.kl_weight = kl_weight
        self.optimizer_kwargs = optimizer_kwargs
        self.current_batch = None
        self.current_outputs = None

    def training_step(self, batch, batch_idx):
        self.current_batch = batch  # for debug
        xout = self.merging_model(batch)
        self.current_outputs = xout  # for debug

        self.merging_model.surrogate_posterior.update_observed(
            batch["rasu_id"], xout["hkl"]
        )
        loss = (
            xout["loss_nll"].mean()
            + self.kl_weight * xout["loss_kl"].mean()
            + xout["scale_kl_div"].mean()
        )
        self.log_dict(
            {
                "loss": loss,
                "NLL": xout["loss_nll"].mean(),
                "KL": xout["loss_kl"].mean(),
                "scale_KL": xout["scale_kl_div"].mean(),
            }
        )
        # for debug
        nan_index = 18815
        for i, z in enumerate(xout["z"][:, nan_index]):
            self.log(f"z/{i}", z)

        norms = grad_norm(self, norm_type=2)
        for name, norm in norms.items():
            self.log(f"grad_norm/{name}", norm)

        return loss

    def validation_step(self, batch, batch_idx):
        xout = self.merging_model(batch)
        val_loss = (
            xout["loss_nll"].mean()
            + self.kl_weight * xout["loss_kl"].mean()
            + xout["scale_kl_div"].mean()
        )
        self.log_dict(
            {
                "val_loss": val_loss,
                "val_NLL": xout["loss_nll"].mean(),
                "val_KL": xout["loss_kl"].mean(),
                "val_scale_KL": xout["scale_kl_div"].mean(),
            }
        )
        return val_loss

    def configure_optimizers(self):
        # renaming arguments for Adam optimizer
        optimzer_kwargs = {
            "lr": self.optimizer_kwargs["learning_rate"],
            "betas": (self.optimizer_kwargs["beta_1"], self.optimizer_kwargs["beta_2"]),
            "eps": self.optimizer_kwargs["adam_epsilon"],
            "amsgrad": self.optimizer_kwargs["amsgrad"],
        }
        opt = Adam(self.merging_model.parameters(), **optimzer_kwargs)
        return opt


class GradientValidator(L.Callback):
    def __init__(self):
        super().__init__()

    def on_before_optimizer_step(self, trainer, pl_module, optimizer):
        # Check all parameters for invalid gradients
        for name, param in pl_module.named_parameters():
            if param.grad is not None:
                if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                    print(f"Invalid gradients found in parameter: {name}")
                    print(
                        f"Gradient stats - min: {param.grad.min()}, max: {param.grad.max()}, mean: {param.grad.mean()}"
                    )
                    print("Dropping into IPython for debugging...")
                    from IPython import embed

                    embed(colors="linux")
                    return


def _group_args(parser):
    # Group arguments from an ArgumentParser into namespaces by their argument groups.
    # This allows accessing arguments by their group name rather than having all arguments
    # in a single flat namespace.
    # Credit to: https://stackoverflow.com/questions/31519997
    import argparse

    args = parser.parse_args()
    arg_groups = {}
    for group in parser._action_groups:
        group_dict = {
            action.dest: getattr(args, action.dest, None)
            for action in group._group_actions
        }
        arg_groups[group.title] = argparse.Namespace(**group_dict)
    return args, arg_groups


def main():
    # ========== parse arguments ==========#
    from abismal_torch.command_line.parser import parser

    args, arg_groups = _group_args(parser)

    L.seed_everything(args.seed)

    # ========== load data ==========#
    from abismal_torch.io.manager import AbismalDataModule

    data = AbismalDataModule(
        args.inputs,
        batch_size=args.batch_size,
        num_workers=args.num_cpus,
        dmin=args.dmin,
        wavelength=args.wavelength,
        rasu_ids=args.rasu_ids,
        persistent_workers=args.persistent_workers,
        pin_memory=args.pin_memory,
    )

    # ========== construct RASU ==========#
    from abismal_torch.symmetry.reciprocal_asu import ReciprocalASU, ReciprocalASUGraph

    rasus = []
    num_rasus = data.num_asus
    for rasu_id in range(num_rasus):
        rasu = ReciprocalASU(
            data.cell[rasu_id],
            data.spacegroup[rasu_id],
            data.dmin,
            anomalous=args.anomalous,
        )
        rasus.append(rasu)
    rac = ReciprocalASUGraph(*rasus)

    # ========== construct model components==========#
    from abismal_torch.scaling import ImageScaler

    scaling_model = ImageScaler(
        **arg_groups["Architecture"].__dict__,
        scaling_kl_weight=args.scale_kl_weight,
    )

    if args.studentt_dof is None:
        from abismal_torch.likelihood import NormalLikelihood

        likelihood = NormalLikelihood()
    else:
        from abismal_torch.likelihood import NormalLikelihood, StudentTLikelihood

        likelihood = StudentTLikelihood(args.studentt_dof)

    from abismal_torch.prior import WilsonPrior

    prior = WilsonPrior(rac, args.normalization_sigma)

    from abismal_torch.surrogate_posterior import FoldedNormalPosterior

    loc_init = prior.distribution().mean()
    scale_init = args.init_scale * loc_init
    surrogate_posterior = FoldedNormalPosterior.from_unconstrained_loc_and_scale(
        rac, loc_init, scale_init, epsilon=args.epsilon
    )

    from abismal_torch.merging import VariationalMergingModel

    merging_model = VariationalMergingModel(
        scaling_model,
        surrogate_posterior,
        prior,
        likelihood,
        mc_samples=args.mc_samples,
        reindexing_ops=args.reindexing_ops,
    )

    # ========== Lightning training ==========#
    model = AbismalLitModule(
        merging_model, arg_groups["Optimizer"].__dict__, kl_weight=args.kl_weight
    )
    callbacks = [
        MTZSaver(out_dir=args.out_dir, save_every_n_epoch=args.save_every_nepochs),
        ModelCheckpoint(dirpath=args.out_dir, filename="model_{epoch:02d}"),
        # PosteriorPlotter(save_every_n_epoch=args.save_every_nepochs),
        GradientValidator(),
    ]

    wandb_logger = WandbLogger(
        project="abismal_torch", save_dir=args.out_dir, name=args.log_run_name
    )
    wandb_logger.watch(surrogate_posterior.distribution, log_freq=1)

    def check_posterior_grad_hook(grad):
        if grad is not None:
            nan_mask = torch.isnan(grad)
            inf_mask = torch.isinf(grad)
            if nan_mask.any():
                nan_indices = nan_mask.nonzero(as_tuple=False).tolist()
                wandb_logger.log_text(
                    key="posterior invalid grads",
                    columns=["nan_indices"],
                    data=[nan_indices],
                )
            if inf_mask.any():
                inf_indices = inf_mask.nonzero(as_tuple=False).tolist()
                wandb_logger.log_text(
                    key="posterior invalid grads",
                    columns=["inf_indices"],
                    data=[inf_indices],
                )
            grad_stats = {
                "posterior loc grad min": grad.min(),
                "posterior loc grad max": grad.max(),
                "posterior loc grad mean": grad.mean(),
            }
            trainer.logger.experiment.log(grad_stats)

    surrogate_posterior.distribution.loc.register_hook(check_posterior_grad_hook)

    trainer = L.Trainer(
        deterministic=True,
        accelerator=args.accelerator,
        min_steps=args.epochs * args.steps_per_epoch,
        max_steps=args.epochs * args.steps_per_epoch,
        default_root_dir=args.out_dir,
        callbacks=callbacks,
        log_every_n_steps=10,
        check_val_every_n_epoch=10,
        logger=wandb_logger,
    )
    trainer.fit(model, data, ckpt_path=args.ckpt_path)


if __name__ == "__main__":
    main()
