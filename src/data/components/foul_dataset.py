import torch
import os
import glob

from PIL import Image
from torch.utils.data import Dataset
from typing import Any, Optional, List, Dict

class Foul_Dataset(Dataset):
    def __init__(self,
                 data_dir: str,
                 transform = None,
                 train: bool=True):
        self.data_dir = data_dir
        self.transform = transform
        self.train = train
        self.img_paths = []
        self.img_labels = []
        self.get_data()

    def __getitem__(self, index):
        img = Image.open(self.img_paths[index])
        img = self.transform(img)

        if self.train:
            return img, torch.tensor(self.img_labels[index], dtype=torch.int64)
        
        return img, self.img_paths[index]
      
    def __len__(self):
        return len(self.img_paths)

    def get_origin_item(self, index):
        return self.img_paths[index], self.img_labels[index]
    
    def get_data(self):
        pass
