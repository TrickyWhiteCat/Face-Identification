import os
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets

from typing import Callable

class FaceNetDataset(datasets.ImageFolder):
    def __init__(self, root:str, transform: Callable = None, target_transform: Callable = None, samples_per_label: int = 1):
        '''Initializes the FaceNet dataset.
        Parameters:
            root (str): The root directory of the dataset.
            transform (Callable, optional): The transform to apply to the images. Defaults to None.
            target_transform (Callable, optional): The transform to apply to the labels. Defaults to None.
            samples_per_label (int, optional): The number of samples to select per label. Defaults to 1.
        '''
        super().__init__(root, transform=transform, target_transform=target_transform)
        self.samples_per_label = samples_per_label

    def __getitem__(self, index: int):
        '''Returns the sample at the given index.
        Parameters:
            index (int): The index of the sample.
        Returns:
            tuple: A tuple of the sample and the label.
        '''
        # Get the image and label
        img, label = super().__getitem__(index)
        # Get all samples with the same label
        same_label_idx = torch.where(torch.tensor(self.targets) == label)[0]
        # Select samples_per_label samples
        selected_idx = torch.randperm(len(same_label_idx))[:self.samples_per_label]
        # Select samples
        selected_samples = torch.index_select(self.data, 0, same_label_idx[selected_idx])
        # Return the selected samples and the label
        return selected_samples, label