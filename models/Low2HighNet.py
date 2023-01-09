from torch import optim, nn
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torchvision.transforms import ToPILImage
from .UNet import UNetModel


class Low2HighNet(pl.LightningModule):

    def __init__(self, lr, ch_in, ch_out):
        super().__init__()
        self.learning_rate = lr
        self.model = UNetModel(ch_in=ch_in, ch_out=ch_out)

    def training_step(self, batch, batch_idx):
        x, x_low, x_mid, x_high, mask = batch
        x_hat = self.model(x_low)

        # LOSS
        loss = nn.functional.mse_loss(x_hat * mask, x_high * mask)

        # LOGGING
        self.log("train/mse", loss)
        if batch_idx % 40 == 0:
            self.log_img(x, x_low, x_high, x_hat, key="train/result")

        return loss

    def validation_step(self, batch, batch_idx):
        x, x_low, x_mid, x_high, mask = batch
        x_hat = self.model(x_low)

        # LOSS
        loss = nn.functional.mse_loss(x_hat * mask, x_high * mask)

        # LOGGING
        self.log("valid/mse", loss, sync_dist=True)
        if batch_idx % 4 == 0:
            self.log_img(x, x_low, x_high, x_hat, key="valid/result")

        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def log_img(self, *xs, key="results"):
        xs = [x.repeat(1, 3, 1, 1) if x.shape[-3] == 1 else x for x in xs]
        xs = [x.add(1).div(2).clamp(0, 1) for x in xs]
        imgs = torch.cat(xs, dim=-2)
        self.logger.log_image(key=key,
                              images=[ToPILImage()(img) for img in imgs])
