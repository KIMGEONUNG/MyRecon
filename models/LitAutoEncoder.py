from torch import optim, nn
import pytorch_lightning as pl


# define the LightningModule
class LitAutoEncoder(pl.LightningModule):

    def __init__(self, lr):
        super().__init__()
        self.learning_rate = lr

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.MaxPool2d(2, 2),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 3, 2, stride=2),
        )

    def training_step(self, batch, batch_idx):
        x, y = batch
        z = self.encoder(x)
        x_hat = self.decoder(z)

        # LOSS
        loss = nn.functional.mse_loss(x_hat, x)

        # LOGGING
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        z = self.encoder(x)
        x_hat = self.decoder(z)

        # LOSS
        loss = nn.functional.mse_loss(x_hat, x)

        # LOGGING
        self.log("valid_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer
