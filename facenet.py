import logging
logging.basicConfig(level=logging.WARNING)

from utils.split_folder import split_folder

import os
from facenet.trainer import train
import torch
from torchvision import models, datasets
from torchvision.transforms import v2 as transforms
from matplotlib import pyplot as plt



SAVE_DIR = 'checkpoints'
DATA_DIR = r'data\lfw-deepfunneled'
SPLIT_DATA_DIR = r'data\lfw-deepfunneled-split'
os.makedirs(SAVE_DIR, exist_ok=True)

embedding_size = 128

epochs = [10, 0, 0]
distance_metric = 'euclidean'
n_classes = 16
n_samples = 4
margin = 1
device = 'cuda'

learning_rate = (5*10**-3, 10**-4)
finetune_learning_rate = (10**-3, 10**-5)

model = models.mobilenet_v3_small(weights = models.MobileNet_V3_Small_Weights.DEFAULT)
model.requires_grad_(False)
model.classifier = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(576, embedding_size))
augment = transforms.Compose([transforms.ToTensor(),
                              transforms.RandomErasing(p=0.4, scale=(0.01, 0.05)),
                              transforms.RandomHorizontalFlip(p=0.2),])
model_transform = models.MobileNet_V3_Small_Weights.DEFAULT.transforms(antialias = True)
transform = transforms.Compose([augment, model_transform])
train_dir, valid_dir = split_folder(DATA_DIR, SPLIT_DATA_DIR, ratio=0.8)
train_dataset = datasets.ImageFolder(train_dir, transform=transform)
valid_dataset = datasets.ImageFolder(valid_dir, transform=transform)
r1 = train(model = model,
           train_dataset = train_dataset,
           distance_metric = distance_metric,
           device=device,
           margin=margin,
           learning_rate=learning_rate[0],
           semihard_negative=True,
           save_file=os.path.join(SAVE_DIR, f'{model.__class__.__name__}_{learning_rate}_{distance_metric}.pth',),
           epochs = epochs[0],
           n_classes=n_classes,
           n_samples = n_samples,
           validation_dataset = valid_dataset,
           end_learning_rate=learning_rate[1],)
model.requires_grad_(True)
r2 = train(model = model,
           train_dataset = train_dataset,
           distance_metric = distance_metric,
           device=device,
           margin=margin,
           learning_rate=finetune_learning_rate[0],
           semihard_negative=True,
           save_file=os.path.join(SAVE_DIR, f'finetune_{model.__class__.__name__}_{finetune_learning_rate}_{distance_metric}.pth',),
           epochs = epochs[1],
           n_classes=n_classes,
           n_samples = n_samples,
           validation_dataset = valid_dataset,
           end_learning_rate=finetune_learning_rate[1],)
# Fine-tune the model
r3 = train(model = model,
           train_dataset = train_dataset,
           distance_metric = distance_metric,
           device=device,
           margin=margin,
           learning_rate=finetune_learning_rate[0],
           semihard_negative=False,
           save_file=os.path.join(SAVE_DIR, f'hard_{model.__class__.__name__}_{finetune_learning_rate}_{distance_metric}.pth',),
           epochs = epochs[2],
           n_classes=n_classes,
           n_samples = n_samples,
           validation_dataset = valid_dataset,
           end_learning_rate=finetune_learning_rate[1],)
record = r1[0] + r2[0] + r3[0]
valid_record = r1[1] + r2[1] + r3[1]

plt.plot(record, label='train')
plt.plot(valid_record, label='valid')
plt.legend()
plt.savefig('facenet.png', dpi=600)
plt.show()