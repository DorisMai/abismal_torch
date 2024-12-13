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
        # TODO
        pass

    def configure_optimizers(self):
        opt = Adam(self.merging_model.parameters(), **self.optimizer_kwargs)
        return opt
