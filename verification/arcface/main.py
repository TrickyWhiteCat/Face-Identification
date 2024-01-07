from trainer import train
import torch
from torchvision import datasets, models
from torchvision.transforms import v2 as transforms
from matplotlib import pyplot as plt
import os
from onnx2torch import convert
from pathlib import Path

from utils.datasets import SubsetWithTransforms
from arcface.head import ArcFaceEmbeddingHead

SAVE_DIR = r'checkpoints/arcface'
DATA_DIR = r"data/lfw-deepfunneled"
os.makedirs(SAVE_DIR, exist_ok=True)

embedding_size = 512

epochs = [10, 50]
scale = 64
batch_size = 16
margin = 0.5
device = 'cuda'

learning_rate = (10**-3 * batch_size, 5*10**-4 * batch_size)
finetune_learning_rate = (5*10**-4 * batch_size, 10**-4 * batch_size)

model = models.mobilenet_v3_small(weights = models.MobileNet_V3_Small_Weights.DEFAULT)
model.requires_grad_(False)
model.classifier = ArcFaceEmbeddingHead(embedding_size, 576)
augment = transforms.Compose([transforms.RandomErasing(p=0.4, scale=(0.02, 0.4)),
                              transforms.RandomHorizontalFlip(p=0.5),])
model_transform = transforms.Compose([transforms.Resize([224, 224], antialias=True),
                                      transforms.Normalize(mean = [0.4392, 0.3831, 0.3424], std=[0.2969, 0.2735, 0.2682])]) # data specific normalization
train_transform = transforms.Compose([transforms.ToTensor(), model_transform])
dataset = datasets.ImageFolder(DATA_DIR)
train_subset, valid_subset = torch.utils.data.random_split(dataset, [0.9, 0.1])
train_dataset = SubsetWithTransforms(train_subset, transform = train_transform)
valid_dataset = SubsetWithTransforms(valid_subset, transform=transforms.Compose([transforms.ToTensor(), model_transform]))

classify_matrix = torch.nn.Parameter(torch.normal(0, 0.01, (len(dataset.classes), embedding_size), device=device))

# Knowledge distillation
onnx_model_path = Path("models", "verification", "onnx", "ms1mv3_r50.onnx")
converted_model = convert(onnx_model_path)
teacher_model = converted_model.eval().to(device)
teacher_transforms = transforms.Compose([transforms.Resize([112, 112], antialias=True),])

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
           teacher_model=teacher_model,
           teacher_weight=0.1,
           teacher_transforms=teacher_transforms,
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
           teacher_model=teacher_model,
           teacher_weight=0.5,
            teacher_transforms=teacher_transforms,
           save_epochs=True)
record = r1[0] + r2[0]
valid_record = r1[1] + r2[1]

plt.plot(record, label='train')
plt.plot(valid_record, label='valid')
plt.legend()
plt.savefig('arcface.png', dpi=600)
plt.show()