import logging
logging.basicConfig(level=logging.INFO)

from typing import Callable

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets
import torch.nn.utils.parametrize as parametrize
from pytorch_metric_learning import losses, miners, distances, reducers, testers

def train(model:nn.Module, train_dataset: datasets.ImageFolder, learning_rate: float = 0.01, epochs: int = 10, save_file:str = None, verbose = True, device: str = 'cpu', validation_dataset: datasets.ImageFolder = None, end_learning_rate = None, save_each_epoch=True, **dataloader_kwargs):
    '''
    Train the model with the given hyperparameters. When being called with `model(x)`, the model must return the embeddings of the input `x`.
    The model will be trained with the FaceNet procedure, which means that the model will be trained with triplet loss and the data loader will be created with `BalancedBatchSampler`.

    Parameters:
    -----------
    model: nn.Module
        The model to be trained.
    dataset: ImageDataset
        The dataset to train the model on.
    learning_rate: float
        Learning rate of the optimizer. Default to 0.001.
    epochs: int
        Number of epochs.
    save_file: str
        Path to save the model's state dict. If None, the model will not be saved.
    verbose: bool
        Whether to print the training progress or not.
    device: str
        Device to run the model on.
    validation_dataset: ImageDataset
        The validation dataset to evaluate the model after each epoch.
    end_learning_rate: float
        The final learning rate of the optimizer. If None, the learning rate will not be changed.
    **dataloader_kwargs:
        Additional keyword arguments to pass to the data loader.
    '''
    train_dataloader = DataLoader(train_dataset, )