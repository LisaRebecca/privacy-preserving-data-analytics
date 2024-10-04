import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()

        self.conv1 = nn.Conv2d(3, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(500, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):  # x.shape()=100,3,32,32
        x = self.conv1(x)  # x.shape()=100,10,28,28
        x = F.max_pool2d(x, 2)  # x.shape()=100,10,14,14
        x = F.relu(x)  # x.shape()=100,10,14,14
        x = self.conv2(x)  # x.shape()=100,20,10,10
        x = self.conv2_drop(x)  # x.shape()=100,20,10,10
        x = F.max_pool2d(x, 2)  # x.shape()=100,20,5,5
        x = F.relu(x)  # x.shape()=100,20,5,5

        x = x.view(-1, 500)
        x = self.fc1(x)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)

        return F.softmax(x, dim=0)


def load_ResNet():
    model = torch.hub.load("pytorch/vision:v0.10.0", "resnet18", pretrained=True)
    from opacus.validators import ModuleValidator

    errors = ModuleValidator.validate(model, strict=False)

    model = ModuleValidator.fix(model)
    ModuleValidator.validate(model, strict=False)
    return model


MODELS = {
    "WideResNet": torch.hub.load(
        "pytorch/vision:v0.10.0", "wide_resnet50_2", pretrained=False
    ),
    "CNN": CNNModel(),
    "ResNet": load_ResNet(),
}
