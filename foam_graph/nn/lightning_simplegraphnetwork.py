import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from foam_graph.nn.graph_network import GraphNetwork


try:
    from pytorch_lightning import LightningModule as PLLightningModule

    no_pytorch_lightning = False
except (ImportError, ModuleNotFoundError):
    PLLightningModule = object
    no_pytorch_lightning = True


class SimpleGraphNetwork(PLLightningModule):
    def __init__(
        self,
        num_node_features,
        num_edge_features,
        num_target_features,
        hidden_channels=128,
        num_hidden=2,
        num_blocks=15,
        learning_rate=1e-3,
    ):
        super().__init__()
        if no_pytorch_lightning:
            raise ModuleNotFoundError(
                "No module named 'pytorch_lightning' found on this machine. "
                "Run 'pip install pytorch_lightning' to install the library."
            )

        self.save_hyperparameters()

        self.model = GraphNetwork(
            num_node_features,
            num_edge_features,
            num_target_features,
            hidden_channels,
            num_hidden,
            num_blocks,
        )

    def forward(self, x):
        out = self.model(x)
        return out

    def training_step(self, batch, batch_idx):
        y = self(batch)
        loss = F.mse_loss(y, batch.y)
        self.log_dict({"train_loss": loss})
        return loss

    def validation_step(self, val_batch, batch_idx):
        y = self(val_batch)
        loss = F.mse_loss(y, val_batch.y)
        metrics = {"val_loss": loss}
        self.log_dict(metrics, prog_bar=True)
        return metrics

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": ReduceLROnPlateau(optimizer),
                "monitor": "val_loss",
            },
        }
