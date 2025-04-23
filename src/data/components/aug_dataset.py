import torch

from torchvision.transforms import transforms
from torch.utils.data import Dataset
from typing import Optional, List
from PIL import Image

from src.data.components.foul_dataset import Foul_Dataset

class DataAugumentation(Dataset):
    def __init__(
        self, 
        dataset, 
        aug_types: Optional[List] = ["horizontal_flip", "vertical_flip", "rotation", "color_jitter", "gaussian_blur"], 
        image_size: Optional[List] = [256, 256]
    )->None:
        super().__init__()
        self.dataset = dataset
        self.aug_types = aug_types
        self.image_size = image_size
        self.img_paths = []
        self.img_labels = []
    
        self.setup()
        self.get_transforms()

    def setup(self):
        for i in range(len(self.dataset)):
            img_path, img_label = self.dataset.get_origin_item(i)
            for aug_type in self.aug_types:
                self.img_paths.append((img_path, aug_type))

            self.img_labels += [img_label]*len(self.aug_types)

    def get_transforms(self):
        self.transforms = {}

        for aug_type in self.aug_types:

            if aug_type == "horizontal_flip":
                self.transforms[aug_type] = transforms.Compose(
                    [
                        transforms.ToTensor(),
                        transforms.Resize((self.image_size[0], self.image_size[1])),
                        transforms.RandomHorizontalFlip(p=1),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                    ]
                )
                continue

            if aug_type == "vertical_flip":
                self.transforms[aug_type] = transforms.Compose(
                    [
                        transforms.ToTensor(),
                        transforms.Resize((self.image_size[0], self.image_size[1])),
                        transforms.RandomVerticalFlip(p=1),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                    ]
                )
                continue

            if aug_type == "rotation":
                self.transforms[aug_type] = transforms.Compose(
                    [
                        transforms.ToTensor(),
                        transforms.Resize((self.image_size[0], self.image_size[1])),
                        transforms.RandomRotation(90),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                    ]
                )
                continue
            
            if aug_type == "color_jitter":
                self.transforms[aug_type] = transforms.Compose(
                    [
                        transforms.ToTensor(),
                        transforms.Resize((self.image_size[0], self.image_size[1])),
                        transforms.ColorJitter(brightness=0.5, contrast=0.5),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                    ]
                )
                continue
            
            if aug_type == "gaussian_blur":
                self.transforms[aug_type] = transforms.Compose(
                    [
                        transforms.ToTensor(),
                        transforms.Resize((self.image_size[0], self.image_size[1])),
                        transforms.GaussianBlur(kernel_size=(3, 3)),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                    ]
                )
                continue
            
    def __len__(self):
        return len(self.img_paths)
    
    def __getitem__(self, index):
        img = Image.open(self.img_paths[index][0])
        img = self.transforms[self.img_paths[index][1]](img)

        return img, torch.tensor(self.img_labels[index], dtype=torch.int64)