import os

from typing import Any, Dict, Optional, Tuple, List
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import transforms

from .constants import LABEL_TO_INT
from src.utils import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)

class FoulDataModule(LightningDataModule):
    def __init__(
        self,
        dataset_classes: Optional[List] = ["Var_Dataset", "SoccerNet_Dataset", "Image_Event_Dataset"],
        data_dirs: Optional[Dict] = {
            "Var_Dataset": "data/kaggle/splited_data",
            "Image_Event": "data/Image_Event",
            "SoccerNet": "data/SoccerNet/mvfouls"
        },
        train_val_test_split: Tuple[float, float, float] = (0.8, 0.1, 0.1),
        split_by_label: Optional[bool] = True,
        batch_size: Optional[int] = 64,
        image_size: Optional[List] = [256, 256], 
        num_workers: Optional[int] = 0,
        pin_memory: Optional[bool] = False,
        train: bool = True   
    ) -> None:
        super().__init__()

        self.save_hyperparameters(logger=False)

        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize((image_size[0], image_size[1])),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]
        )

        self.dataset_classes = dataset_classes
        self.data_dirs = data_dirs
        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

        self.batch_size_per_device = batch_size
        self.train = train

    @property
    def num_classes(self) -> int:
        return len(LABEL_TO_INT.keys())
    
    def setup(self, stage: Optional[str] = None) -> None:
        for dataset_class in self.dataset_classes:
            if dataset_class == "My_Dataset":
                dataset_obj = eval(dataset_class)
                self.data_train = dataset_obj(
                    data_dir = os.path.join(self.data_dirs[dataset_class], "train"),
                    transform = self.transform,
                    train=self.train
                )
                self.data_val = dataset_obj(
                    data_dir = os.path.join(self.data_dirs[dataset_class], "valid"),
                    transform = self.transform,
                    train=self.train
                )

                self.data_test = dataset_obj(
                    data_dir = os.path.join(self.data_dirs[dataset_class], "test"),
                    transform = self.transform,
                    train=self.train
                )
        log.info(f"Length of training set: {len(self.data_train)}")
        log.info(f"Length of validation set: {len(self.data_val)}")
        log.info(f"Length of testing set: {len(self.data_test)}")

    def train_dataloader(self) -> DataLoader[Any]:
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True
        )
    
    def test_dataloader(self) -> DataLoader[Any]:
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True
        )
    
    def val_dataloader(self) -> DataLoader[Any]:
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True
        )