import torch 

from torch import nn
from typing import Optional, Tuple, List
from collections import OrderedDict
from src.data.constants import LABEL_TO_INT

class VarCNN(nn.Module):
    def __init__(
            self,
            image_size: Optional[List] = [256, 256]
        ):
        super(VarCNN, self).__init__()
        self.image_size = image_size
        self.model = self._get_model()

    def forward(self, image: torch.Tensor):
        return self.model(image)

    def _get_model(self) -> nn.Module:
        initial = nn.Sequential(
            nn.Conv2d(3, 64, 5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(64, 64, 5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 64, 5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 64, 5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        filter = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1, dilation=2),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 16, 3, padding=1, dilation=2),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 16, 3, padding=1, dilation=2),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )

        classifer = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features= int(16*self.image_size[0]*self.image_size[1]/ 2**16), 
                      out_features=len(LABEL_TO_INT.keys()))
        )

        model = nn.Sequential(OrderedDict([
            ("initial", initial),
            ("filter", filter),
            ("classifer", classifer)
        ]))

        return model