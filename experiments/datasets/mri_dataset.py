import numpy as np
import torch
from torch.utils.data import Dataset
import json
import os
from scipy import ndimage

LABEL_MAP = {
    "sMCI": 0,
    "pMCI": 1,
}

#TODO THIS FILE WILL NEED CHANGING ONCE I CREATE MY CUSTOM pMCI DATASET

def normalize(volume):
    """Z-score using foreground (nonzero) voxels; background stays 0."""
    fg = volume > 0
    out = np.zeros(volume.shape, dtype=np.float32)
    out[fg] = (volume[fg] - volume[fg].mean()) / volume[fg].std()
    return out

class MRIDataset(Dataset):
    def __init__(self, json_path, image_dir, transform=None, cache_dir=None):

        #Read in all json data
        with open(json_path, 'r') as file:
            all_data = json.load(file)

        #Initialize variables and cache directory
        self.image_dir = image_dir
        self.transform = transform
        self.cache_dir = cache_dir
        if cache_dir is not None:
            os.makedirs(cache_dir, exist_ok=True)

        #Append MRI images with matching json labels
        self.data = []
        for item in all_data:
            image_path = os.path.join(image_dir, f"{item['id']}.npz")
            if os.path.exists(image_path):
                self.data.append(item)
            else:
                print(f"Skipped missing image: {item['id']}.npz")

        #Sort data
        self.data = sorted(self.data, key=lambda x: x["id"])

    def _load_image(self, id_):
        cache_path = os.path.join(self.cache_dir, f"{id_}.npy") if self.cache_dir else None
        if cache_path and os.path.exists(cache_path):
            return np.load(cache_path)
        image = normalize(np.load(self._image_path(id_))["image_mr"])
        if cache_path:
            np.save(cache_path, image)
        return image

    def __len__(self):
        return len(self.data)

    def _image_path(self, id_):
        return os.path.join(self.image_dir, f"{id_}.npz")

    def __getitem__(self, idx):
        item = self.data[idx]
        image = torch.from_numpy(self._load_image(item["id"]))[None]  # (1, D, H, W) #is this correct?
        if self.transform:
            image = self.transform(image)

        return image, LABEL_MAP[item["label"]]
    