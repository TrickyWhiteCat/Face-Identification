from trainer import train
import torch
from torchvision import datasets, models
from torchvision.transforms import v2 as transforms
from matplotlib import pyplot as plt
import os

from utils.datasets import SubsetWithTransforms
from head import ArcFaceEmbeddingHead

SAVE_DIR = r'checkpoints/arcface'
DATA_DIR = r"data/lfw-deepfunneled"
os.makedirs(SAVE_DIR, exist_ok=True)

embedding_size = 128

epochs = [10, 20]
scale = 64
batch_size = 128
margin = 0.5
device = 'cuda'

learning_rate = (10**-3 * batch_size, 5*10**-4 * batch_size)
finetune_learning_rate = (5*10**-4 * batch_size, 10**-5 * batch_size)

model = models.mobilenet_v3_small(weights = models.MobileNet_V3_Small_Weights.DEFAULT)
model.requires_grad_(False)
model.classifier = ArcFaceEmbeddingHead(embedding_size, 576)

augment = transforms.Compose([transforms.RandomErasing(p=0.6, scale=(0.02, 0.4)),
                              transforms.RandomResizedCrop(size = [224, 224], scale=(0.7, 1.0), ratio=(0.75, 1.25),antialias=True),
                              transforms.RandomHorizontalFlip(p=0.5),])
model_transform = transforms.Compose([transforms.Resize([224, 224], antialias=True),
                                      transforms.Normalize(mean = [0.5313, 0.4263, 0.3748], std=[0.2873, 0.2552, 0.2492])]) # Dataset norm
train_transform = transforms.Compose([transforms.ToTensor(), augment, model_transform])

dataset = datasets.ImageFolder(DATA_DIR)
train_subset, valid_subset = torch.utils.data.random_split(dataset, [0.9, 0.1])
train_dataset = SubsetWithTransforms(train_subset, transform = train_transform)
valid_dataset = SubsetWithTransforms(valid_subset, transform=transforms.Compose([transforms.ToTensor(), model_transform]))
classify_matrix = torch.nn.Parameter(torch.normal(0, 0.01, (len(dataset.classes), embedding_size), device=device))

r1 = train(model = model,
           train_dataset = train_dataset,
           embedding_size=embedding_size,
           device=device,
           margin=margin,
           learning_rate=learning_rate[0],
           save_file=os.path.join(SAVE_DIR, f'{model.__class__.__name__}.pth',),
           epochs = epochs[0],
           batch_size = batch_size,
           scale = scale,
           validation_dataset = valid_dataset,
           end_learning_rate=learning_rate[1],
           classify_matrix=classify_matrix,
           save_epochs=True)
model.requires_grad_(True)
r2 = train(model = model,
           train_dataset = train_dataset,
           embedding_size=embedding_size,
           device=device,
           margin=margin,
           learning_rate=finetune_learning_rate[0],
           save_file=os.path.join(SAVE_DIR, f'finetune_{model.__class__.__name__}.pth',),
           epochs = epochs[1],
           batch_size = batch_size,
           scale = scale,
           validation_dataset = valid_dataset,
           end_learning_rate=finetune_learning_rate[1],
           classify_matrix=classify_matrix,
           save_epochs=True)
record = r1[0] + r2[0]
valid_record = r1[1] + r2[1]

plt.plot(record, label='train')
plt.plot(valid_record, label='valid')
plt.legend()
plt.savefig('arcface.png', dpi=600)
plt.show()