import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from model import InceptionV2, TripletLoss, train_model
from dataset import FaceNetDataset
from torch.utils.data import random_split


# Define transforms
train_transform_color = transforms.Compose([
    transforms.ToTensor(),
    transforms.RandomRotation(degrees=10),
    transforms.RandomHorizontalFlip(p=0.2),
])

valid_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

# Create datasets and data loaders
root_folder = '/kaggle/input/ms1m-retinaface-t1-subset/imgs_subset'
dataset = FaceNetDataset(root=root_folder)

train_dataset_color, valid_dataset_color, _ = random_split(
    dataset, [int(0.8 * len(dataset)), int(0.1 * len(dataset)), len(dataset) - int(0.9 * len(dataset))]
)

train_loader_color = DataLoader(train_dataset_color, batch_size=64, shuffle=True, num_workers=4)
valid_loader_color = DataLoader(valid_dataset_color, batch_size=64, shuffle=False, num_workers=4)

# Train the model
resume_checkpoint = 'new'
train_model(train_loader_color, valid_loader_color, 
            save_file="InceptionV2.pth", 
            wandb_project="FaceVeri_InceptionV2", 
            train_transform = train_transform_color, 
            early_stop_patience=20, epoch = 300, 
            resume_checkpoint=resume_checkpoint)