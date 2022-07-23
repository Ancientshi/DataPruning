import torch
import torch.nn as nn
class MyNet(nn.Module):
    def __init__(self, num_classes=10):
        super(MyNet, self).__init__()
        self.feature_extraction = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=96, kernel_size=6, stride=1, padding=2, bias=False),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0),
            nn.Conv2d(in_channels=96, out_channels=128, kernel_size=4, stride=1, padding=1, bias=False),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=32, kernel_size=4, stride=1, padding=1, bias=False),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0),
            nn.Conv2d(in_channels=32, out_channels=128, kernel_size=6, stride=1, padding=0, bias=False),
            nn.LeakyReLU(inplace=True),
        )

        self.classifier = nn.Sequential(
            #nn.Dropout(0.5),
            nn.Linear(in_features=128, out_features=num_classes)
        )

    def forward(self, x):
        x = self.feature_extraction(x)
        x=x.view(x.size(0),128)
        x = self.classifier(x)
        return x