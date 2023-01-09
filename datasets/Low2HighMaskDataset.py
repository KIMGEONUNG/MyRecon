from torchvision.transforms import (ToTensor, Compose, Resize, CenterCrop,
                                    InterpolationMode, Grayscale)
from typing import Tuple, Any, Optional, Callable
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np


class Low2HighMaskDataset(Dataset):

    def __init__(
        self,
        path_index: str,
        size: int,
    ):
        super().__init__()
        self.transform_img = Compose([
            ToTensor(),
            Resize(size),
            CenterCrop(size),
            lambda x: x * 2 - 1,
        ])
        self.transform_mask = Compose([
            ToTensor(),
            Resize(size, interpolation=InterpolationMode.NEAREST),
            CenterCrop(size),
        ])

        with open(path_index, "r") as f:
            paths = f.read().splitlines()

        # Define Path
        path_img = [path + ".JPEG" for path in paths]
        path_mask = [
            path.replace("train", "train.mask").replace("valid", "valid.mask")
            + ".npy" for path in paths
        ]

        self.samples = list(zip(path_img, path_mask))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        path_img, path_mask = self.samples[index]

        x = self.loader_img(path_img)
        x = self.transform_img(x)

        mask = self.loader_mask(path_mask)
        mask = self.transform_mask(mask)

        x_low, x_mid, x_high = self.extract_low2high(x, mask)

        return x, x_low, x_mid, x_high, mask

    def extract_low2high(self, x, mask):
        x_ = x + 1
        x_ = x_ * mask
        x_ = x_.mean(dim=(1, 2))
        order = torch.argsort(x_)
        x_low, x_mid, x_high = x[order]
        return x_low[None, ...], x_mid[None, ...], x_high[None, ...]

    def loader_img(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def loader_mask(self, path):
        return np.load(path)
