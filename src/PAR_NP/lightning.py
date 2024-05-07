import lightning as L
import neuralprocesses as nps
import torch

from PAR_NP.models import construct_model


class LitNP(L.LightningModule):
    def __init__(self, model, model_params, lr=1e-3):
        super().__init__()
        self.model = construct_model(model_name, **model_params)
        self.lr = lr

    def loss(self, x_context, y_context, x_target, y_target):
        return -torch.mean(
            nps.loglik(
                self.model,
                x_context.float(),
                y_context.float(),
                x_target.float(),
                y_target.float(),
                normalise=True,
            )
        )

    def training_step(self, batch, batch_idx):
        loss = self.loss(*batch)
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), self.lr)
