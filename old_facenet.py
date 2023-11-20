import os

import numpy as np
import torch
from facenet.model import FaceNet
from torch import nn
from torchvision import transforms
from torchvision import models
from matplotlib import pyplot as plt

import logging
logging.basicConfig(level=logging.ERROR, filename="facenet.log", filemode="w")

torch.cuda.empty_cache()
# Test if FaceNet works on pretrained ResNet18
extractor = models.mobilenet_v3_small(weights = models.MobileNet_V3_Small_Weights.DEFAULT)
transform = models.MobileNet_V3_Small_Weights.DEFAULT.transforms()
extractor.classifier = nn.Identity()
extractor.requires_grad_(False)
'''
# Test on newly created extractor
extractor = nn.Sequential(nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
                          nn.ReLU(),
                          nn.MaxPool2d(kernel_size=2, stride=1),
                          nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
                          nn.ReLU(),
                          nn.MaxPool2d(kernel_size=2, stride=1),)
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Resize((128, 128))])
'''
save_dir = 'checkpoints'
os.makedirs(save_dir, exist_ok=True)


epochs = 5
learning_rate = (0.01, 0.001)

model = FaceNet(extractor,
                data_dir=r"data\lfw-deepfunneled-split\train",
                transform=transform,
                distance_metric='euclidean',
                device='cuda')
r1 = model.autotrain(learning_rate=learning_rate[0],
                     n_classes=4,
                     n_samples=8,
                     epochs=epochs,
                     semihard_negative=True)
r2 = []

model.requires_grad_(True)
# Fine-tune the model
r2 = model.autotrain(learning_rate=learning_rate[1],
                     n_classes=4,
                     n_samples=8,
                     epochs=epochs,
                     semihard_negative=False,
                     save_file=os.path.join(save_dir, f'facenet_{epochs}epochs_{learning_rate}_cosine.pth'))
record = r1 + r2

plt.plot(record)
plt.legend()
plt.savefig('facenet.png', dpi=600)
plt.show()