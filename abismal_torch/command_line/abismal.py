from typing import Optional

import lightning as L
from torch.optim import Adam


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
        self.merging_model = merging_model
        self.kl_weight = kl_weight
        self.optimizer_kwargs = optimizer_kwargs

    def training_step(self, batch, batch_idx):
        xout = self.merging_model(batch)
        loss = xout["loss_nll"] + self.kl_weight * xout["loss_kl"]
        self.log_dict({"loss": loss, "NLL": xout["loss_nll"], "KL": xout["loss_kl"]})
        return loss

    def validation_step(self, batch, batch_idx):
        xout = self.merging_model(batch)
        val_loss = xout["loss_nll"] + self.kl_weight * xout["loss_kl"]
        self.log_dict(
            {
                "val_loss": val_loss,
                "val_NLL": xout["loss_nll"],
                "val_KL": xout["loss_kl"],
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
    from abismal_torch.io.manager import MTZDataModule

    mtz_file = args.inputs[0]
    data = MTZDataModule(
        mtz_file,
        batch_size=args.batch_size,
        num_workers=args.num_cpus,
        dmin=args.dmin,
        wavelength=args.wavelength,
    )

    # ========== construct RASU ==========#
    from abismal_torch.symmetry.reciprocal_asu import ReciprocalASU, ReciprocalASUGraph

    rasus = []
    for dataset in data.dataset.datasets:
        rasu = ReciprocalASU(
            dataset.cell,
            dataset.spacegroup,
            dataset.dmin,
            anomalous=args.anomalous,
        )
        rasus.append(rasu)
    rac = ReciprocalASUGraph(*rasus)

    # ========== construct model components==========#
    from abismal_torch.scaling import ImageScaler

    scaling_model = ImageScaler(**arg_groups["Architecture"].__dict__)

    from abismal_torch.likelihood import StudentTLikelihood

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
    opt_args = arg_groups["Optimizer"].__dict__
    model = AbismalLitModule(merging_model, opt_args, kl_weight=args.kl_weight)

    trainer = L.Trainer(
        deterministic=True,
        accelerator="cpu",
        max_epochs=args.epochs,
        max_steps=args.steps_per_epoch * args.epochs,
    )
    trainer.fit(model, data)


if __name__ == "__main__":
    main()
