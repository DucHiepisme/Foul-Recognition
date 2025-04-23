import os
import glob

from typing import Optional

from ..constants import LABEL_TO_INT
from .foul_dataset import Foul_Dataset

LABEL = {
    "Fouls": "foul",
    "Clean_Tackles": "no foul"
}

class Var_Dataset(Foul_Dataset):
    def __init__(self,
                 data_dir: Optional[str] = "data/kaggle/VAR",
                 transform = None,
                 train: bool = True):
        super(Var_Dataset, self).__init__(data_dir, transform, train)

    def get_data(self):
        if self.train:
            for k, v in LABEL.items():
                img_folder = os.path.join(self.data_dir, k)
                img_paths = glob.glob(os.path.join(img_folder, "*.jpg"))
                for path in img_paths:
                    self.img_paths.append(path)
                    self.img_labels.append(LABEL_TO_INT[v])
        else:
            img_paths = glob.glob(os.path.join(self.data_dir, "*.jpg"))
            for path in img_paths:
                self.img_paths.append(path)