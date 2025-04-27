import re
import torchvision

from pathlib import Path
from torch.utils.data import Dataset
from torchvision.io import read_image
from torchvision.transforms import Compose


import logging

log = logging.getLogger(__name__)


class BrainMRIDataset(Dataset):
    categories = ["no", "pred", "yes"]
    read_mode = torchvision.io.ImageReadMode.GRAY

    def __init__(
        self,
        path_data: str,
        split: str,
        n_skip: int,
        n_samples: int,
        transform: Compose,
    ):
        super(BrainMRIDataset, self).__init__()
        assert split in ["Training", "Testing"]
        self.paths = self.get_paths(path_data)
        self.length = min(len(self.paths), n_samples)
        self.n_skip = n_skip
        self.transform = transform
        log.info(f"Data transformations for {split=}: {transform}")

    def map_idx(self, idx):
        return self.n_skip + idx

    def get_paths(self, path):
        paths_c = [Path(path) / c for c in self.categories]
        paths = sum(
            [
                sorted(
                    p.glob("*.jpg"),
                    key=lambda x: int(re.search(r"\d+", Path(x).stem).group()),
                )
                for p in paths_c
            ],
            [],
        )
        return paths

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        index = self.map_idx(index)
        path = self.paths[index]
        img = read_image(str(path), mode=self.read_mode) / 255
        return self.transform(img) if self.transform else img
