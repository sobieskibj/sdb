from pathlib import Path
from torch.utils.data import Dataset
from torchvision.io import read_image
from torchvision.transforms import Compose


import logging
log = logging.getLogger(__name__)


class CelebAHQDataset(Dataset):
    """
    torch.utils.data.Dataset for CelebA-HQ. Assumes that images are stored as:

    celebahq/
        <split>/
            img_<img_id>.png

    and that there are 30 000 images in total with 27000 / 3000 per training / validation split.
    """
    

    def __init__(
            self, 
            path_data: str, 
            split: str,
            n_samples: int,
            transform: Compose):
        
        super(CelebAHQDataset, self).__init__()
        
        self.split = split
        self.paths = self.get_paths(path_data, split)
        self.length = min(len(self.paths), n_samples)
        self.transform = transform
        log.info(f"Data transformations for {split=}: {transform}")

    def get_paths(self, path, split):
        paths = [Path(path) / split / f"img_{idx}.png" for \
                 idx in range(30_000 if split == "train" else 3_000)]
        return paths


    def __len__(self):
        return self.length


    def __getitem__(self, index):
        path = self.paths[index]
        img = read_image(str(path)) / 255
        return self.transform(img) if self.transform else img

