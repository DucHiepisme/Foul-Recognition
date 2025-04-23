import torch
import os
import glob
import zipfile

from typing import Any, Optional, List, Dict

from ..constants import LABEL_TO_INT
from .foul_dataset import Foul_Dataset

LABEL = {
    "Cards": "foul",
    "Center": "no foul",
    "Corner": "no foul",
    "Free-Kick": "no foul",
    "Left": "no foul",
    "Penalty": "no foul",
    "Red-Cards": "foul",
    "Right": "no foul",
    "Tackle": "no foul",
    "To-Subtitue": "no foul",
    "Yellow-Cards": "foul"
}

class Image_Event_Dataset(Foul_Dataset):
    def __init__(self,
                 data_dir: Optional[str] = "data/Image_Event",
                 transform = None):
        super(Image_Event_Dataset, self).__init__(data_dir, transform)

    def get_data(self):
        pass
