import pytest
import torch

from src.models.components import *
from src.data.constants import LABEL_TO_INT

@pytest.mark.parametrize("model", [VarCNN, Resnet18, Resnet34, Resnet50, Resnet101, Resnet152])
def test_foul_model(model)->None:
    _img_size = 512
    img = torch.ones((1, 3, _img_size, _img_size))

    _model = model(image_size=(_img_size, _img_size))
    
    ligit = _model(img)

    batch_size = ligit.size()[0]
    cls = ligit.size()[-1]
    print(ligit.size())
    assert batch_size == 1
    assert cls == len(LABEL_TO_INT.keys())