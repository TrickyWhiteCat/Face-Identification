from arcface import train
import torch
from torch import nn
from torchvision import datasets
from torchvision.transforms import v2 as transforms

class ExtractorWithFC(torch.nn.Module):
    def __init__(self, extractor, n_classes, fc_features = 128):
        super().__init__()
        self.extractor = extractor
        self.fc = torch.nn.Linear(fc_features, n_classes)
    def forward(self, x):
        x = self.extractor(x)
        x = self.fc(torch.flatten(x, 1))
        return x
    
class ConvReLU(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size = 3, stride = 1, padding = 1, batch_norm = False, activation = None):
        super().__init__()
        self.conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.relu = torch.nn.ReLU()
        if batch_norm:
            self.batch_norm = torch.nn.BatchNorm2d(out_channels)
        else:
            self.batch_norm = torch.nn.Identity()
        if activation is None:
            self.activation = torch.nn.Identity()
        else:
            self.activation = activation
    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        x = self.batch_norm(x)
        x = self.activation(x)
        return x
    
def main():
    # A simple extractor to be used as tester
    extractor = nn.Sequential(*[ConvReLU(1, 64, 3, batch_norm = True, activation = torch.nn.MaxPool2d(2)),
                                ConvReLU(64, 128, 3, batch_norm = True, activation = nn.AdaptiveAvgPool2d((1, 1))),
    ])
    model = ExtractorWithFC(extractor, 10, 128)
    optimizer = torch.optim.Adam(model.parameters())
    train(model, 'cpu', torch.utils.data.DataLoader(datasets.MNIST(root="MNIST", train = True, transform=transforms.ToTensor())), optimizer, 1)

if __name__ == '__main__':
    main()  
