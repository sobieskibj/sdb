from pathlib import Path
from torchvision.datasets import MNIST
import torchvision.transforms.functional as TF

class MNISTDataset(MNIST):
    '''
    Subclass of the MNIST dataset provided by torchvision with additional functionalities.
    '''

    def __init__(
            self, 
            root: str|Path, 
            n_samples: int,
            train: bool, 
            transform: callable, 
            target_transform: callable, 
            download: bool):
        
        super(MNISTDataset, self).__init__(
            root=root,
            train=train,
            transform=transform, 
            target_transform=target_transform, 
            download=download
        )

        self.n_samples = min(n_samples, 60_000 if train else 10_000)


    def __len__(self):
        return self.n_samples


    def __getitem__(self, idx):

        # we do not utilize labels so they are simply omitted here
        image, _ = super(MNISTDataset, self).__getitem__(idx)

        # resizing to 32x32 to not coincide with network implementation
        image = TF.resize(image, 32).repeat(3, 1, 1)

        return image # NOTE: we assume [0, 1] range