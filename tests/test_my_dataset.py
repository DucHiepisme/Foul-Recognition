import pytest
import torch

from torchvision.transforms import transforms

from src.data.components import My_Dataset

def test_var_dataset()->None:
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize((256, 256)),
            transforms.Normalize((0.1307, 0.1307, 0.1307), (0.3081, 0.3081, 0.3081))
        ]
    )
    my_dataset = My_Dataset(transform=transform)

    assert my_dataset.__len__() > 0
    print("Len dataset: ", len(my_dataset))
    img, label = my_dataset.__getitem__(2)
    assert img.size()[1] == 256
    assert label.dtype == torch.int64
