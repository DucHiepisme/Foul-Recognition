import torch

from torchvision.transforms import transforms

from src.data.components.aug_dataset import DataAugumentation
from src.data.components.var_dataset import Var_Dataset

def test_aug_dataset():
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize((256, 256)),
            transforms.Normalize((0.1307, 0.1307, 0.1307), (0.3081, 0.3081, 0.3081))
        ]
    )
    var_dataset = Var_Dataset(transform=transform)

    aug_types = ["horizontal_flip", "vertical_flip", "rotation", "color_jitter", "gaussian_blur"]
    aug_dataset = DataAugumentation(var_dataset)

    assert len(aug_dataset) == len(aug_types) * len(var_dataset)
    
    img, label = aug_dataset.__getitem__(2)
    assert img.size()[1] == 256
    assert label.dtype == torch.int64