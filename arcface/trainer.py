import logging
logging.basicConfig(level=logging.INFO)

from typing import Callable

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets
from pytorch_metric_learning import losses

def train(model:nn.Module, embedding_size, train_dataset: datasets.ImageFolder, learning_rate: float = 0.01, batch_size: int = 1, epochs: int = 10, margin = 0.5, scale = 64, save_file:str = None, save_epochs = False, verbose = True, device: str = 'cpu', validation_dataset: datasets.ImageFolder = None, end_learning_rate = None, loss_fn = None, **dataloader_kwargs):
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
    logger = logging.getLogger("ArcFaceTrainer")
    model.to(device)
    model.train()
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, **dataloader_kwargs)
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    gamma = (end_learning_rate / learning_rate) ** (1 / epochs)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
    logger.info(f"Learning rate scheduler is created with the gamma {gamma}. Expected end learning rate is {end_learning_rate}.")
    if loss_fn is None:
        loss_fn = losses.ArcFaceLoss(num_classes=len(train_dataset.classes), embedding_size=embedding_size, margin=margin, scale=scale).to(device)
        logger.info(f"Loss function is created with num_classes={len(train_dataset.classes)}, {embedding_size=}, {margin=}, {scale=}.")
    loss_optimizer = optim.SGD(loss_fn.parameters(), lr=learning_rate) # The loss function involves a trainable tensor
    loss_scheduler = optim.lr_scheduler.ExponentialLR(loss_optimizer, gamma=gamma)
    logger.info(f"Reusing loss function")

    record_loss = []
    record = 0
    record_valid_loss = []
    for epoch in range(epochs):
        for idx, (x, y) in enumerate(train_dataloader):
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            loss_optimizer.zero_grad()
            embeddings = model(x)
            loss = loss_fn(embeddings, y)
            loss.backward()
            optimizer.step()
            loss_optimizer.step()
            if verbose:
                record += loss.item()
                print(f"Epoch {epoch + 1}/{epochs} -- batch {idx + 1}/{len(train_dataloader)} -- loss: {loss.item():.4f}", end  = '\r')
        scheduler.step()
        loss_scheduler.step()
        if save_file is not None:
            if save_epochs:
                torch.save(model.state_dict(), save_file[:-4] + f"_epoch_{epoch + 1}.pth")
            else:
                torch.save(model.state_dict(), save_file)
        record_loss.append(record / len(train_dataloader))
        record = 0
        if validation_dataset is None:
            logger.warning("No validation dataset is provided. Skip evaluation.")
            continue
        model.eval()
        with torch.no_grad():
            val_loss = 0
            valid_dataloader = DataLoader(validation_dataset, batch_size=batch_size)
            for x, y in valid_dataloader:
                x = x.to(device)
                y = y.to(device)
                embeddings = model(x)
                val_loss += loss_fn(embeddings, y).item()
            val_loss /= len(valid_dataloader)
        record_valid_loss.append(val_loss)
        model.train()
    logger.info("Training finished.")
    return record_loss, record_valid_loss