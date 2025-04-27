from pathlib import Path
from torch.utils.data import Dataset
from torchvision.io import read_image
from torchvision.transforms import Compose


import logging
log = logging.getLogger(__name__)


class DIV2KDataset(Dataset):
    """
    torch.utils.data.Dataset for DIV2K. Assumes that images are stored as:
    """
    

    def __init__(
            self, 
            path_data: str, 
            split: str,
            n_samples: int,
            transform: Compose):
        
        super(DIV2KDataset, self).__init__()
        
        self.split = split
        self.paths = self.get_paths(path_data, split)
        self.length = min(len(self.paths), n_samples)
        self.transform = transform
        log.info(f"Data transformations for {split=}: {transform}")

    def get_paths(self, path, split):
        
        if split == "valid":
            paths = [Path(path) / f"{idx:04}.png" for idx in range(801, 901)]

        elif split == "train":
            paths = [Path(path) / f"{idx:04}.png" for idx in range(1, 801)]

        else:
            raise ValueError(f"Unrecognized value {split} for split.")
        
        return paths


    def __len__(self):
        return self.length


    def __getitem__(self, index):
        path = self.paths[index]
        img = read_image(str(path)) / 255
        return self.transform(img) if self.transform else img

