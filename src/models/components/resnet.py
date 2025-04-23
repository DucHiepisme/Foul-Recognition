import torch

from torch import nn
from torchvision import models
from typing import Optional, List

from src.data.constants import LABEL_TO_INT

class Resnet18(nn.Module):
    def __init__(
            self,
            image_size: Optional[List] = [256, 256]
        ):
        super(Resnet18, self).__init__()
        self.image_size = image_size
        self.model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.replace_final_layer()
        
    def forward(self, image: torch.Tensor):
        return self.model(image)
    
    def replace_final_layer(self):
    
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, len(set(LABEL_TO_INT.values())))

class Resnet34(nn.Module):
    def __init__(
            self,
            image_size: Optional[List] = [256, 256]
        ):
        super(Resnet34, self).__init__()
        self.image_size = image_size
        self.model = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
        self.replace_final_layer()
        
    def forward(self, image: torch.Tensor):
        return self.model(image)
    
    def replace_final_layer(self):
    
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, len(LABEL_TO_INT.keys()))

class Resnet50(nn.Module):
    def __init__(
            self,
            image_size: Optional[List] = [256, 256]
        ):
        super(Resnet50, self).__init__()
        self.image_size = image_size
        self.model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.replace_final_layer()
        
    def forward(self, image: torch.Tensor):
        return self.model(image)
    
    def replace_final_layer(self):
    
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, len(LABEL_TO_INT.keys()))

class Resnet101(nn.Module):
    def __init__(
            self,
            image_size: Optional[List] = [256, 256]
        ):
        super(Resnet101, self).__init__()
        self.image_size = image_size
        self.model = models.resnet101(weights=models.ResNet101_Weights.DEFAULT)
        self.replace_final_layer()
        
    def forward(self, image: torch.Tensor):
        return self.model(image)
    
    def replace_final_layer(self):
    
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, len(LABEL_TO_INT.keys()))

class Resnet152(nn.Module):
    def __init__(
            self,
            image_size: Optional[List] = [256, 256]
        ):
        super(Resnet152, self).__init__()
        self.image_size = image_size
        self.model = models.resnet152(weights=models.ResNet152_Weights.DEFAULT)
        self.replace_final_layer()
        
    def forward(self, image: torch.Tensor):
        return self.model(image)
    
    def replace_final_layer(self):
    
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, len(LABEL_TO_INT.keys()))