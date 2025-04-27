from tqdm import tqdm
import torch
import pydicom
import pandas as pd
from pathlib import Path
from torch.utils.data import Dataset


import logging
log = logging.getLogger(__name__)


class RSNAIHDataset(Dataset):


    def __init__(
            self, 
            path_data: str, 
            path_metadata: str, 
            n_samples: int, 
            transform: object, 
            expected_size: int,
            n_skip: int):
        
        super(RSNAIHDataset, self).__init__()

        self.path_data = Path(path_data)
        self.metadata = self.get_metadata(Path(path_metadata), n_samples)
        self.transform = transform
        self.expected_size = expected_size
        self.length = min(n_samples, len(self.metadata))
        self.n_skip = n_skip


    def map_idx(self, x):
        return x + self.n_skip


    def get_metadata(self, path_metadata, n_samples):
        
        def get_id_col(path):
            # load as df
            df = pd.read_csv(path)
            
            # replace label suffix
            df.ID = df.ID.str.extract(r'^([^_]+_[^_]+)')

            return pd.Index(df.ID).unique().tolist()

        # modify to get filtered metadata path
        path_filtered = path_metadata.with_stem(f"{path_metadata.stem}_filtered")

        if path_filtered.exists():
            # if it exists, we simply load and convert it
            metadata = get_id_col(path_filtered)
            log.info("Loading filtered metadata")    

        else:
            # otherwise, we iterate over images and exclude those that have improper shape
            
            # get original ids
            ids = get_id_col(path_metadata)[:n_samples]

            log.info("Filtering images")

            for idx in tqdm(ids[:]):
                # find patient dicom dir
                patient_dicom_dir = self.path_data / f"{idx}.dcm"

                # load dicom and convert to array
                dicom_data = pydicom.dcmread(patient_dicom_dir)    
                image = dicom_data.pixel_array            

                # check shape
                if image.shape[0] != 512 or image.shape[1] != 512:
                    # remove if invalid
                    ids.remove(idx)

            # save filtered ids and use as metadata
            pd.DataFrame(ids, columns=["ID"]).to_csv(path_filtered)
            metadata = ids

        return metadata


    def __len__(self):
        return self.length
    

    def __getitem__(self, idx):
        
        # map index to include skipping
        idx = self.map_idx(idx)

        if isinstance(idx, slice):
            print("encountered slice")
            start = idx.start
            stop = idx.stop
            step = idx.step

            if start is None:
                start = 0
            if stop is None:
                stop = len(self.metadata)
            if step is None:
                step = 1

            image_list = []

            for i in range(start, stop, step):
                image = self.__getitem__(i)
                image_list.append(image)

            return torch.concat(image_list, dim=0)

        # extract patient_id
        patient_id = self.metadata[idx]
        
        # get path to patient dicom dir and load the dicom
        patient_dicom_dir = self.path_data / f"{patient_id}.dcm"
        dicom_data = pydicom.dcmread(patient_dicom_dir)
        
        # get pixel array and add intercept
        assert float(dicom_data.RescaleSlope) == 1.0, 'RescaleSlope is not 1.0'
        image = dicom_data.pixel_array + float(dicom_data.RescaleIntercept)

        # apply optional transforms
        if self.transform:
            image = self.transform(image)
        else:
            image = torch.tensor(image).float()
            image.unsqueeze_(0) # add channel dimension
        
        # handle cases where image is not the expected size
        if self.expected_size == 512 and (image.shape[1] != 512 or image.shape[2] != 512):
            # handle this case more gracefully, e.g., by skipping or logging
            # return self.__getitem__((idx + 1) % len(self.metadata))
            print(f"Warning: Image at index {idx} is not 512x512, returning image at index {idx+1}")
            return self.__getitem__(idx + 1)
        
        if self.expected_size == 256 and (image.shape[1] != 256 or image.shape[2] != 256):
            # handle size check for reconstructed images
            raise ValueError(f"Image at index {idx} is not 256x256")
        
        # perform downsampling if the expected size is 512
        if self.expected_size == 512:
            image = torch.nn.functional.avg_pool2d(image, kernel_size=2)
            image = torch.clip(image, -1000, 2000)
        
        # clip to -1000 to 2000
        image = torch.clip(image, -1000, 2000)

        # assert that resolution is correct
        assert image.shape[1] == image.shape[2] == 256

        # return HU scaled to attenuation
        return _HU_to_attenuation(image)
    

def _HU_to_attenuation(image, scale_only=False):
    """
    scale_only == False:
    μ = (HU + 1000) * 0.1 / 1000 

    scale_only == True:
    μ = HU * 0.1 / 1000 
    """
    if scale_only:
        return (image) * 0.1 / 1000.0
    else:
        return (image + 1000.0) * 0.1 / 1000.0
    

def _attenuation_to_HU(image, scale_only=False):
    """
    scale_only == False:
    μ = (HU + 1000) * 0.1 / 1000 

    scale_only == True:
    μ = HU * 0.1 / 1000 
    """
    if scale_only:
        return (image * 1000.0 / 0.1)
    else:
        return (image * 1000.0 / 0.1) - 1000.0