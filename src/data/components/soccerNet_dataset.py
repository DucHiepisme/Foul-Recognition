import torch
import os
import glob

from PIL import Image
from torch.utils.data import Dataset
from typing import Any, Optional, List, Dict

class SoccerNet_Dataset(Dataset):
    def __init__(self,
                 data_dir: Optional[str] = "",
                 transform = None):
        pass

    def __getitem__(self, index):
        pass

    def __len__(self):
        pass
