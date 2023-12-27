import logging
import math

from arcface.loss import ArcFace
logging.basicConfig(level=logging.INFO)

from typing import Callable

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets
from .knowledge_distillation import losses

def train(model:nn.Module, embedding_size, train_dataset: datasets.ImageFolder, learning_rate: float = 0.01, batch_size: int = 1, epochs: int = 10, margin = 0.5, scale = 64, save_file:str = None, save_epochs = False, verbose = True, device: str = 'cpu', validation_dataset: datasets.ImageFolder = None, end_learning_rate = None, classify_matrix = None, teacher_model = None, teacher_weight = 0.5, teacher_transforms = nn.Identity(), **dataloader_kwargs):
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

    if classify_matrix is None:
        classify_matrix = torch.nn.Parameter(torch.FloatTensor(len(train_dataset.classes), embedding_size))
        torch.nn.init.xavier_uniform_(classify_matrix)
    penalty = ArcFace(s=scale, margin=margin,)
    loss_fn = nn.CrossEntropyLoss()
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, **dataloader_kwargs)

    optimizer = optim.SGD([{'params': model.parameters()}, {'params': classify_matrix}], lr=learning_rate)
    # Set step size so that the learning rate will be reduced to end_learning_rate at the last epoch.
    gamma = 0.5
    step_size =  math.log(end_learning_rate / learning_rate, gamma)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size = step_size, gamma=gamma)
    logger.info(f"Learning rate scheduler is created with the gamma {gamma}. Expected end learning rate is {end_learning_rate}.")

    record_loss = []
    record = 0
    record_valid_loss = []
    for epoch in range(epochs):
        for idx, (x, y) in enumerate(train_dataloader):
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            embeddings = model(x) # Normalized
            logits = nn.functional.linear(nn.functional.normalize(embeddings), nn.functional.normalize(classify_matrix))
            penalized = penalty(logits, y)
            if teacher_model is not None:
                teacher_embeddings = teacher_model(teacher_transforms(x))
                if teacher_embeddings.shape != embeddings.shape:
                    raise ValueError(f"Expected the shape of the teacher's embeddings to be {embeddings.shape}, but got {teacher_embeddings.shape}.")
                teacher_logits = nn.functional.linear(nn.functional.normalize(teacher_embeddings), nn.functional.normalize(classify_matrix))
                teacher_penalized = penalty(teacher_logits, y)
                teacher_loss = losses.SoftTarget(T=2)(penalized, teacher_penalized)
            else:
                teacher_loss = 0
                teacher_weight = 0
            loss = (1 - teacher_weight) * loss_fn(penalized, y) + teacher_weight * teacher_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5) # Reuse the same value as in the original implementation
            optimizer.step()
            if verbose:
                record += loss.item()
                print(f"Epoch {epoch + 1}/{epochs} -- batch {idx + 1}/{len(train_dataloader)} -- loss: {loss.item():.4f}", end  = '\r')
        scheduler.step()
        if save_file is not None:
            savepath = save_file[:-4] + f"_epoch_{epoch + 1}.pth"
            torch.save({"model_state_dict": model.state_dict(),
                        "classify_matrix": classify_matrix}, savepath)
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
                logits = nn.functional.linear(nn.functional.normalize(embeddings), nn.functional.normalize(classify_matrix))
                penalized = penalty(logits, y)
                loss = loss_fn(penalized, y)
                val_loss += loss.item()
            val_loss /= len(valid_dataloader)
        record_valid_loss.append(val_loss)
        model.train()
    logger.info("Training finished.")
    return record_loss, record_valid_loss
