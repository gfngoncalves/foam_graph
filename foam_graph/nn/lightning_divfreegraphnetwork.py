import torch
import torch.nn.functional as F
from foam_graph.nn.lightning_simplegraphnetwork import SimpleGraphNetwork
from foam_graph.utils.physics_calcs import div
import numpy as np


class DivFreeGraphNetwork(SimpleGraphNetwork):
    def __init__(
        self,
        num_node_features,
        num_edge_features,
        num_target_features,
        hidden_channels=128,
        num_hidden=2,
        num_blocks=15,
        learning_rate=1e-3,
        loss_div_weight=1,
        loss_div_samples=1,
    ):
        super().__init__(
            num_node_features,
            num_edge_features,
            num_target_features,
            hidden_channels,
            num_hidden,
            num_blocks,
            learning_rate,
        )

    def _div_loss(self, y, batch):
        idx_div_samples = np.random.default_rng().choice(
            len(y), self.hparams.loss_div_samples
        )
        div_losses = []
        for j, i in enumerate(idx_div_samples):
            div_loss_i = div(
                y,
                batch,
                i,
                self.trainer.datamodule.edge_normalize.to(self.device),
                self.trainer.datamodule.target_normalize.to(self.device),
            )
            div_losses.append(div_loss_i)
        div_losses = torch.stack(div_losses)
        div_loss = F.mse_loss(div_losses, torch.zeros_like(div_losses))
        div_loss /= self.hparams.loss_div_samples
        return div_loss

    def backward(self, loss, optimizer, optimizer_idx):
        loss.backward(retain_graph=True)

    def training_step(self, batch, batch_idx):
        batch.edge_attr.requires_grad_()
        y = self(batch)
        loss = F.mse_loss(y, batch.y)

        div_loss = self._div_loss(y, batch)
        loss += self.hparams.loss_div_weight * div_loss

        self.log_dict({"train_loss": loss, "div_loss": div_loss})
        return loss

    def on_validation_model_eval(self, *args, **kwargs):
        super().on_validation_model_eval(*args, **kwargs)
        torch.set_grad_enabled(True)

    def validation_step(self, val_batch, batch_idx):
        val_batch.edge_attr.requires_grad_()
        y = self(val_batch)
        loss = F.mse_loss(y, val_batch.y)

        div_loss = self._div_loss(y, val_batch)
        loss += self.hparams.loss_div_weight * div_loss

        metrics = {"val_loss": loss, "div_loss": div_loss}
        self.log_dict(metrics, prog_bar=True)
        return metrics
