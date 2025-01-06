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
        opt = Adam(self.merging_model.parameters(), **self.optimizer_kwargs)
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


def _get_rasu():
    import gemmi

    from abismal_torch.symmetry.reciprocal_asu import ReciprocalASU, ReciprocalASUGraph

    rasu_params = {
        "spacegroups": [gemmi.SpaceGroup(19), gemmi.SpaceGroup(4)],
        "dmins": [9.1, 8.8],
        "cell": gemmi.UnitCell(10.0, 20.0, 30.0, 90.0, 90.0, 90.0),
    }
    anomalous = False
    rasu1 = ReciprocalASU(
        rasu_params["cell"],
        rasu_params["spacegroups"][0],
        rasu_params["dmins"][0],
        anomalous,
    )
    rasu2 = ReciprocalASU(
        rasu_params["cell"],
        rasu_params["spacegroups"][1],
        rasu_params["dmins"][1],
        anomalous,
    )
    rag = ReciprocalASUGraph(rasu1, rasu2)
    return rag


def main():
    from abismal_torch.command_line.parser import parser

    args, arg_groups = _group_args(parser)

    from abismal_torch.scaling import ImageScaler

    scaling_model = ImageScaler(**arg_groups["Architecture"].__dict__)

    from abismal_torch.likelihood import StudentTLikelihood

    likelihood = StudentTLikelihood(args.studentt_dof)

    rac = _get_rasu()

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

    opt_args = arg_groups["Optimizer"].__dict__
    model = AbismalLitModule(merging_model, opt_args, kl_weight=args.kl_weight)

    print("done construction")
    print(args)


if __name__ == "__main__":
    main()
