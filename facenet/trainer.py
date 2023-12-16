'''A function to train any model using FaceNet procedure.'''
import logging
from typing import Callable

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets
from pytorch_metric_learning.samplers import MPerClassSampler
from pytorch_metric_learning.losses import TripletMarginLoss
from pytorch_metric_learning.miners import TripletMarginMiner

logging.basicConfig(level=logging.INFO, filename='facenet_trainer.log', filemode='w')

def train(model:nn.Module, train_dataset: datasets.ImageFolder, margin: int = 1, learning_rate: float = 0.01, n_samples = 8, batch_size = 64, epochs: int = 10, semihard_negative:bool = True, save_file:str = None, save_epochs = False, verbose = True, device: str = 'cpu', validation_dataset: datasets.ImageFolder = None, end_learning_rate = None,**dataloader_kwargs):
    '''
    Train the model with the given hyperparameters. When being called with `model(x)`, the model must return the embeddings of the input `x`.
    The model will be trained with the FaceNet procedure, which means that the model will be trained with triplet loss and the data loader will be created with `BalancedBatchSampler`.

    Parameters:
    -----------
    model: nn.Module
        The model to be trained.
    dataset: ImageDataset
        The dataset to train the model on.
    distance_metric: str, Callable
        The distance metric to use when comparing face embeddings. If Callable, must take two tensors and an optional parameter `dim` to specify the dimension to perform calculation.
    learning_rate: float
        Learning rate of the optimizer. Default to 0.001.
    n_classes: int
        Number of classes in a batch. This parameter is used to create the data loader.
    n_samples: int
        Number of samples per class in a batch. This parameter is used to create the data loader.
    epochs: int
        Number of epochs.
    semihard_negative: bool
        Whether to use semihard negative mining or not.
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
    logger = logging.getLogger("FaceNetTrainer")
    logger.info("Training started.")
    # Create data loader
    train_dataloader = DataLoader(train_dataset, sampler=MPerClassSampler(train_dataset.targets, m=n_samples, length_before_new_iter=len(train_dataset), batch_size=batch_size), batch_size = batch_size, **dataloader_kwargs)
    valid_dataloader = DataLoader(validation_dataset, sampler=MPerClassSampler(validation_dataset.targets, m=n_samples, length_before_new_iter=len(validation_dataset), batch_size=batch_size), batch_size = batch_size, **dataloader_kwargs)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    logger.info(f"Optimizer is created with the learning rate {learning_rate}.")
    gamma = (end_learning_rate / learning_rate) ** (1 / epochs)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
    logger.info(f"Learning rate scheduler is created with the gamma {gamma}. Expected end learning rate is {end_learning_rate}.")
    # Create loss function
    miner = TripletMarginMiner(margin=margin, type_of_triplets="semihard" if semihard_negative else "hard")
    loss_fn = TripletMarginLoss(margin=margin, swap=True, smooth_loss=False)
    logger.info(f"Loss function is created with the margin {margin}.")
    record_loss = []
    record = 0
    record_valid_loss = []
    # Train
    model.train()
    model.to(device)
    for epoch in range(epochs):
        for idx, (x, y) in enumerate(train_dataloader):
            # Zero the gradients
            optimizer.zero_grad()
            embeddings = model(x.to(device))
            hard_pairs = miner(embeddings, y)
            loss = loss_fn(embeddings, y, hard_pairs)
            loss.backward()
            optimizer.step()
            if device == 'cuda':
                gpu_usage = torch.cuda.memory_usage(device=device)
                if gpu_usage >= 90:
                    logger.warning(f"High GPU usage: {gpu_usage}%.")
            if verbose:
                record += loss.item()
                print(f"Epoch {epoch + 1}/{epochs} -- batch {idx + 1}/{len(train_dataloader)} -- loss: {loss.item():.4f}{'' if 'cuda' not in device else f' -- GPU usage: {gpu_usage}%'}", end  = '\r')
        scheduler.step()
        # Save model
        if save_file is not None:
            if save_epochs:
                torch.save(model.state_dict(), save_file[:-4] + f"_epoch_{epoch + 1}.pth")
            else:
                torch.save(model.state_dict(), save_file)
            logger.info(f"Model is saved to {save_file}.")
        # Evaluate
        record_loss.append(record / len(train_dataloader))
        record = 0
        if validation_dataset is None:
            logger.warning("No validation dataset is provided. Skip validation.")
            continue # Skip validation if no validation dataset is provided
        model.eval()
        with torch.no_grad():
            valid_loss = 0
            for idx, (x, y) in enumerate(valid_dataloader):
                valid_loss += loss_fn(torch.nn.functional.normalize(model(x.to(device)), p=2, dim=1), y.to(device)).item()
            valid_loss /= len(valid_dataloader)
            logger.info(f"Epoch {epoch + 1}/{epochs} -- validation loss: {valid_loss:.4f}")
        record_valid_loss.append(valid_loss)
    logger.info("Training finished.")
    return record_loss, record_valid_loss