import pytest
import torch

from torchvision.transforms import transforms

from src.data.components import Var_Dataset
from src.utils.utils import random_split

def count_label(data):
    count = 0
    for i in range(len(data)):
        if data.__getitem__(i)[-1].item() == 0:
            count +=  1

    print("Label 0: %d, label 1: %d", (count, len(data) - count))

@pytest.mark.parametrize("split_by_label", [True, False])
def test_random_split(split_by_label):
    lengths = (0.8, 0.1, 0.1)
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize((256, 256)),
            transforms.Normalize((0.1307, 0.1307, 0.1307), (0.3081, 0.3081, 0.3081))
        ]
    )
    var_dataset = Var_Dataset(transform=transform)

    train_set, val_set, test_set = random_split(
        dataset=var_dataset,
        lengths=lengths,
        generator=torch.Generator().manual_seed(42),
        split_by_label=split_by_label
    )

    assert len(train_set) + len(test_set) + len(val_set) == len(var_dataset)

    if split_by_label:
        print("")
        count_label(train_set)
        count_label(val_set)
        count_label(test_set)