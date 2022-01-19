import timm
import torch.nn as nn


class BaseModel(nn.Module):
    def __init__(self, cls, model_name="resnet50", pretrained=True):
        super(BaseModel, self).__init__()
        self.base = timm.create_model(model_name, pretrained)
        self.base.fc = nn.Linear(self.base.fc.in_features, cls)

    def forward(self, x):
        output = self.base(x)
        return output
