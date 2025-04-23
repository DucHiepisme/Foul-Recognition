import pytest
import torch

from src.data.foul_datamodule import FoulDataModule

@pytest.mark.parametrize("batch_size", [32, 64])
@pytest.mark.parametrize("split_by_label", [True, False])
def test_foul_datamodule(batch_size: int, split_by_label: bool):
    dataset_classes = ["Var_Dataset"]
    data_module = FoulDataModule(
        split_by_label=split_by_label,
        dataset_classes=dataset_classes, 
        batch_size=batch_size)
    
    data_module.setup()

    assert data_module.train_dataloader() and data_module.val_dataloader() and data_module.test_dataloader()

    batch = next(iter(data_module.train_dataloader()))
    img, label = batch
    assert img.size()[0] == batch_size
    assert img.dtype == torch.float32
    assert label.dtype == torch.int64