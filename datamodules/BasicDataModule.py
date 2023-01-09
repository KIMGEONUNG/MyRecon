import pytorch_lightning as pl
# from ..utils import instantiate_from_config
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose, ToTensor, Resize, CenterCrop


class BasicDataModule(pl.LightningDataModule):

    def __init__(
        self,
        path_train: str,
        path_valid: str,
        size: int,
        config_dataloader_train: str,
        config_dataloader_valid: str,
    ):
        super().__init__()
        self.path_train = path_train
        self.path_valid = path_valid
        self.config_dataloader_train = config_dataloader_train
        self.config_dataloader_valid = config_dataloader_valid

    def prepare_data(self):
        pass

    def setup(self, stage: str):

        t = Compose([
            ToTensor(),
            Resize(256),
            CenterCrop(256),
            lambda x: x * 2 - 1,
        ])

        if stage == "fit" or stage is None:
            self.train = ImageFolder(self.path_train, transform=t)
            self.val = ImageFolder(self.path_valid, transform=t)
        else:
            raise

    def train_dataloader(self):
        return DataLoader(self.train, **self.config_dataloader_train)

    def val_dataloader(self):
        return DataLoader(self.val, **self.config_dataloader_valid)
