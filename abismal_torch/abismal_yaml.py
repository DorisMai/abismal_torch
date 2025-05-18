from typing import Optional, Sequence, Union

import gemmi
import lightning as L
import torch
from lightning.pytorch.utilities import grad_norm

from abismal_torch.callbacks import MTZSaver
from abismal_torch.io.manager import MTZDataModule
from abismal_torch.merging import VariationalMergingModel
from abismal_torch.symmetry.reciprocal_asu import ReciprocalASU, ReciprocalASUGraph


class AbismalLitModule(L.LightningModule):
    def __init__(
        self,
        num_asus: int,
        cell: Union[
            gemmi.UnitCell,
            Sequence[gemmi.UnitCell],
            Sequence[float],
            Sequence[Sequence[float]],
        ],
        spacegroup: Union[
            gemmi.SpaceGroup, Sequence[gemmi.SpaceGroup], str, Sequence[str]
        ],
        dmin: float,
        anomalous: bool,
        scaling_model: torch.nn.Module,
        likelihood: torch.nn.Module,
        prior_config: dict,
        surrogate_posterior_config: dict,
        mc_samples: Optional[int] = 1,
        reindexing_ops: Optional[Sequence[str]] = None,
        kl_weight: Optional[float] = 1.0,
    ):
        """
        Args:
            merging_model (torch.nn.Module): Variational merging model.
            kl_weight (float, optional): KL divergence weight. Defaults to 1.0.
        """
        super().__init__()
        self._rac = self._setup_rac(num_asus, cell, spacegroup, dmin, anomalous)
        self._prior = self._setup_prior(self._rac, prior_config)
        self._surrogate_posterior = self._setup_surrogate_posterior(
            self._rac, surrogate_posterior_config
        )
        self.merging_model = VariationalMergingModel(
            scaling_model,
            self._surrogate_posterior,
            self._prior,
            likelihood,
            mc_samples=mc_samples,
            reindexing_ops=reindexing_ops,
        )
        self.kl_weight = kl_weight
        self.current_batch = None
        self.current_outputs = None

    def _setup_rac(
        self,
        num_asus: int,
        cell: Union[gemmi.UnitCell, Sequence[gemmi.UnitCell]],
        spacegroup: Union[gemmi.SpaceGroup, Sequence[gemmi.SpaceGroup]],
        dmin: float,
        anomalous: bool,
    ):
        if isinstance(cell, Sequence):
            assert len(cell) == num_asus
        else:
            cell = [cell] * num_asus
        if isinstance(spacegroup, Sequence):
            assert len(spacegroup) == num_asus
        else:
            spacegroup = [spacegroup] * num_asus

        rasus = []
        for rasu_id in range(num_asus):
            rasu = ReciprocalASU(
                cell[rasu_id],
                spacegroup[rasu_id],
                dmin,
                anomalous=anomalous,
            )
            rasus.append(rasu)
        rac = ReciprocalASUGraph(*rasus)
        return rac

    def _setup_prior(self, rac: ReciprocalASUGraph, prior_args: dict):
        # construct prior according to "class_path" and "kwargs" in prior_args
        module_path, class_name = prior_args["class_path"].rsplit(".", 1)
        module = __import__(module_path, fromlist=[class_name])
        prior_class = getattr(module, class_name)
        prior_kwargs = prior_args["kwargs"]
        prior = prior_class(rac, **prior_kwargs)
        return prior

    def _setup_surrogate_posterior(
        self, rac: ReciprocalASUGraph, surrogate_posterior_args: dict
    ):
        module_path, class_name = surrogate_posterior_args["class_path"].rsplit(".", 1)
        module = __import__(module_path, fromlist=[class_name])
        surrogate_posterior_class = getattr(module, class_name)
        loc_init = self._prior.distribution().mean()
        scale_init = surrogate_posterior_args["kwargs"]["init_scale"] * loc_init
        surrogate_posterior = (
            surrogate_posterior_class.from_unconstrained_loc_and_scale(
                rac,
                loc_init,
                scale_init,
                epsilon=surrogate_posterior_args["kwargs"]["epsilon"],
            )
        )
        return surrogate_posterior

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
    
    def test_step(self, batch, batch_idx):
        xout = self.merging_model(batch)
        test_loss = (
            xout["loss_nll"].mean()
            + self.kl_weight * xout["loss_kl"].mean()
            + xout["scale_kl_div"].mean()
        )
        self.log_dict(
            {
                "test_loss": test_loss,
                "test_NLL": xout["loss_nll"].mean(),
                "test_KL": xout["loss_kl"].mean(),
                "test_scale_KL": xout["scale_kl_div"].mean(),
            }
        )
        return test_loss

    def configure_optimizers(self):
        optimizer_class = self.hparams.optimizer.class_path
        optimizer_args = self.hparams.optimizer.init_args
        return optimizer_class(self.merging_model.parameters(), **optimizer_args)


from lightning.pytorch.cli import LightningCLI


class MyCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        parser.link_arguments("data.dmin", "model.dmin")
        parser.link_arguments("data.anomalous", "model.anomalous")
        parser.link_arguments("data.num_asus", "model.num_asus", apply_on="instantiate")
        parser.link_arguments("data.cell", "model.cell", apply_on="instantiate")
        parser.link_arguments(
            "data.spacegroup", "model.spacegroup", apply_on="instantiate"
        )
        parser.link_arguments(
            "trainer.default_root_dir", "trainer.logger.init_args.save_dir"
        )
        parser.add_argument("--ckpt_path", type=str, default=None) # only if run=False for MyCLI

        # configure forced callbacks for MTZSaver
        parser.add_lightning_class_args(MTZSaver, "mtz_output")
        # ================================= Note on callback args =================================
        # If using run=True for LightningCLI with subcommand "fit", default values of MTZSaver can 
        # only be overwritten via command line flags but not yaml files. This issue seems absent in
        # instantiation only mode.
        # ====================================== End of note ======================================
        parser.set_defaults(
            {"mtz_output.save_every_n_epoch": 1, "mtz_output.out_dir": None}
        )
        parser.link_arguments("trainer.default_root_dir", "mtz_output.out_dir")


def main():
    import os

    config_dir = "abismal_torch/command_line/configs"
    # cli = MyCLI(
    #     AbismalLitModule,
    #     MTZDataModule,
    #     parser_kwargs={
    #         "fit": {
    #             "default_config_files": [
    #                 os.path.join(config_dir, "data_config.yaml"),
    #                 os.path.join(config_dir, "training_config.yaml"),
    #                 os.path.join(config_dir, "merging_model_config.yaml"),
    #             ],
    #         }
    #     },
    # )

    # Instantiation only mode
    cli = MyCLI(
        AbismalLitModule,
        MTZDataModule,
        run=False,
        parser_kwargs={
            "default_config_files": [
                os.path.join(config_dir, "data_config.yaml"),
                os.path.join(config_dir, "training_config.yaml"),
                os.path.join(config_dir, "merging_model_config.yaml"),
            ],
        }
    )
    cli.trainer.fit(cli.model, cli.datamodule)
    cli.trainer.test(cli.model, cli.datamodule)


if __name__ == "__main__":
    main()
