from torchvision.datasets import ImageFolder
from torchvision.transforms import Grayscale, ToTensor
from typing import Tuple, Any, Optional, Callable

from torch.utils.data import Dataset
from PIL import Image
import numpy as np


class Low2HighMask(Dataset):

    def __init__(
        self,
        index_file,
        trainsform_img=None,
        trainsform_mask=None,
    ):
        super().__init__()
        self.transform_img = trainsform_img
        self.transform_mask = trainsform_mask

        with open(index_file, "r") as f:
            paths = f.read().splitlines()
        # ImagePath
        path_img = [path + ".JPEG" for path in paths]
        # MaskPath
        path_mask = [
            path.replace("train", "train.mask") + ".npy" for path in paths
        ]
        self.samples = list(zip(path_img, path_mask))

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        path_img, path_mask = self.samples[index]

        x = self.loader_img(path_img)
        mask = self.loader_mask(path_mask)

        if self.transform_img is not None:
            x = self.transform_img(x)
        if self.transform_img is not None:
            mask = self.transform_img(mask)

        return None

    def loader_img(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def loader_mask(self, path):
        return np.load(path)


if __name__ == "__main__":
    # d = Low2HighMask('datasets/birds_imagenet/train/')
    d = Low2HighMask('dataset_index/train_birds_vivid_mask.txt')
    print(d[0])
