import sys
sys.path.append("..")
import pytorch_lightning as pl
from utils import instantiate_from_config
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose, ToTensor, Resize, CenterCrop


class DataModule(pl.LightningDataModule):

    def __init__(
        self,
        config_dataset_train: str,
        config_dataset_valid: str,
        config_dataloader_train: str,
        config_dataloader_valid: str,
    ):
        super().__init__()
        self.config_dataset_train = config_dataset_train
        self.config_dataset_valid = config_dataset_valid
        self.config_dataloader_train = config_dataloader_train
        self.config_dataloader_valid = config_dataloader_valid

    def prepare_data(self):
        pass

    def setup(self, stage: str):
        self.train = instantiate_from_config(self.config_dataset_train)
        self.valid = instantiate_from_config(self.config_dataset_valid)

    def train_dataloader(self):
        return DataLoader(self.train, **self.config_dataloader_train)

    def val_dataloader(self):
        return DataLoader(self.valid, **self.config_dataloader_valid)
