'''A function to train any model using FaceNet procedure.'''
import logging
from typing import Callable
import wandb
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets

from facenet.sampler import BalancedBatchSampler

logging.basicConfig(level=logging.INFO, filename='facenet_trainer.log', filemode='w')

'''A function to train any model using FaceNet procedure.'''
def __distance_metric_factory(distance_metric: str|Callable):
    '''Returns the distance metric function.
    Parameters:
        distance_metric (str, Callable): The distance metric to use when comparing face embeddings.
        The distance metric if Callable, must take two tensors and an optional parameter `dim` to specify the dimension to perform calculation.
    Returns:
        Callable: The distance metric function.
    '''
    logger = logging.getLogger("DistanceMetricFactory")
    if distance_metric == 'euclidean':
        return lambda x1, x2, dim=-1: torch.norm(x1 - x2, dim=dim)
    elif distance_metric == 'cosine':
        return lambda x1, x2, dim=-1: 1 - torch.nn.functional.cosine_similarity(x1, x2, dim=dim)
    elif callable(distance_metric):
        try: # Check if the metric function takes 2 tensors and an optional parameter `dim`
            distance_metric(torch.tensor([1, 2]), torch.tensor([3, 4])) # Test with default dim
            distance_metric(torch.tensor([1, 2]), torch.tensor([3, 4]), dim=0) # Test with dim=0
        except TypeError:
            err_msg = f"Distance metric function must take 2 tensors and an optional parameter `dim` to specify the dimension to perform calculation."
            logger.error(err_msg)
            raise TypeError(err_msg)
        return distance_metric
    else:
        err_msg = f"Unknown distance metric: {distance_metric}"
        logger.error(err_msg)
        raise ValueError(err_msg)

def __select_triplets(model, mini_batch: torch.Tensor, distance_metric: Callable, semihard_negative:bool = True, device: str = 'cpu'):
    '''Selects triplets for training based on the method described in the FaceNet paper. Given a mini-batch, first the function will embed the images using the model, then it will select triplets based on the distance between the embeddings. For each sample in the mini-batch, first it is used as an anchor and then a hard positive and a semi-hard negative sample is selected. A hard positive is the sample with the same label as the anchor that is the furthest away from the anchor. A semi-hard negative is the closest sample with a different label than the anchor but still further away than the positive sample within the margin.

    Parameters:
    -----------
        model (nn.Module): The model to be trained.
        mini_batch (torch.Tensor): The mini-batch of images with the shape (N, C, H, W)
        labels (torch.Tensor): The labels of the mini-batch.
        distance_metric (Callable): A function that takes 2 tensors and returns the distance between them.
        semihard_negative (bool): Whether to use semi-hard negative mining or not.
        device (str): Device to run the model on.

    Returns:
        tuple(torch.Tensor): A tuple of the anchor, positive and negative images.
    '''
    logger = logging.getLogger("TripletSelector")
    # Since this function is passed as the `collate_fn` parameter to the data loader, the mini-batch is a list of tuples (image, label).
    # First, separate the images and labels into 2 tensors
    images, labels = zip(*mini_batch)
    images = torch.stack(images) # (N, C, H, W)
    labels = torch.tensor(labels) # (N)
    logger.info(f"Mini-batch is separated into images with the shape {images.shape} and labels with the shape {labels.shape}.")
    '''
    # Shuffle mini-batch
    indices = torch.randperm(images.shape[0])
    images = images[indices]
    labels = labels[indices]
    logger.info(f"Mini-batch is shuffled.")'''
    
    # Move mini-batch to device
    images = images.to(device)
    labels = labels.to(device)
    model = model.to(device)
    logger.info(f"Mini-batch is moved to {device}.")
    # Embed images
    embeddings = torch.nn.functional.normalize(model.forward(images))
    logger.info(f"Batch is embedded into {embeddings.shape}.")
    with torch.no_grad():
        # Select triplets
        # Use 2 methods to calculate per-sample distances to all other samples in the mini-batch:
        # 1. Use vectorization and broadcasting to calculate all distances at once (fast but memory intensive)
        # 2. Use for-loop to calculate distances one-by-one (slow but memory efficient)
        # First try the vectorized implementation, if it fails (usually due to memory issues) then use the for-loop
        # Vectorized implementation
        logger.info(f"Trying vectorized implementation.")
        try:
            # Each row in the embeddings matrix represents a sample in the mini-batch (embedding vectors)
            # The embedding matrix is reshaped into (N, 1, embedding_size) and (1, N, embedding_size)
            # which then are both broadcasted to (N, N, embedding_size) before calculating the distance,
            # results in a tensor of shape (N, N)
            distance = distance_metric(embeddings.unsqueeze(1), embeddings.unsqueeze(0), dim=-1)
        except RuntimeError as e:
            logger.error(f"Vectorized implementation failed: {e}")
            # Clear CUDA cache
            torch.cuda.empty_cache()
            # For-loop implementation
            logger.info(f"Trying for-loop implementation.")
            distance = torch.zeros((embeddings.shape[0], embeddings.shape[0]), device=device)
            for i in range(embeddings.shape[0]):
                for j in range(embeddings.shape[0]):
                    distance[i, j] = distance_metric(embeddings[i], embeddings[j])

        max_distance = distance.max()
        min_distance = (distance + (distance == 0).float() * 10e6).min()
        logger.info(f"Distance matrix is calculated with the shape {distance.shape}.")
        logger.info(f"Max distance calculated is {max_distance} and min distance calculated is {min_distance}.")
        # Create mask for positive and negative samples
        positive_mask = labels.unsqueeze(1) == labels.unsqueeze(0) # positive_mask[i, j] = 1 if i and j have the same label, 0 otherwise
        negative_mask = ~positive_mask
        # Select hard positive samples
        # For each anchor, select the furthest positive sample
        positive_distance, positive_idx = torch.max(distance * positive_mask.float(), dim=1)
        positive = torch.gather(embeddings, 0, positive_idx.unsqueeze(1).repeat(1, embeddings.shape[1]))
        logger.info(f"Positive samples are selected with the shape {positive.shape}.")
        if semihard_negative:
            # Select semi-hard negative samples
            further_than_positive = distance > positive_distance.unsqueeze(1) # A Mask
            neg_further_than_positive = further_than_positive * negative_mask # Another mask
            # For each anchor, select the closest negative sample that is further away than the positive sample
            negative_distance, negative_idx = torch.min(distance * neg_further_than_positive.float() + (~neg_further_than_positive).long() * 10e6, dim=1)
        else:
            # Select hard negative samples
            # For each anchor, select the closest negative sample
            negative_distance, negative_idx = torch.min(distance * negative_mask.float() + positive_mask.long() * 10e6 , dim=1)
        negative = torch.gather(embeddings, 0, negative_idx.unsqueeze(1).repeat(1, embeddings.shape[1]))
        logger.info(f"Negative samples are selected with the shape {negative.shape}.")
    return embeddings, positive, negative

def train(model:nn.Module, train_dataset: datasets.ImageFolder, distance_metric: str|Callable = 'euclidean', margin: int = 1, learning_rate: float = 0.01, n_classes = 8, n_samples = 4, epochs: int = 10, semihard_negative:bool = True, save_file:str = None, verbose = True, device: str = 'cpu', validation_dataset: datasets.ImageFolder = None, end_learning_rate = None, save_each_epoch=True, **dataloader_kwargs):
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
    distance_metric = __distance_metric_factory(distance_metric)
    # Create data loader
    if semihard_negative:
        collate_fn = lambda x: __select_triplets(model, x, distance_metric, semihard_negative=True, device=device)
    else:
        collate_fn = lambda x: __select_triplets(model, x, distance_metric, semihard_negative=False, device=device)
    train_dataloader = DataLoader(train_dataset,
                             collate_fn=collate_fn,
                             batch_sampler=BalancedBatchSampler(train_dataset, n_classes=n_classes, n_samples=n_samples),
                             **dataloader_kwargs)
    logger.info(f"Data loader is created with {n_classes} classes, {n_samples} per classes and epochs {epochs}.")
    # Create optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    logger.info(f"Optimizer is created with the learning rate {learning_rate}.")
    gamma = (end_learning_rate / learning_rate) ** (1 / epochs)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
    logger.info(f"Learning rate scheduler is created with the gamma {gamma}. Expected end learning rate is {end_learning_rate}.")
    # Create loss function
    loss_fn = nn.TripletMarginWithDistanceLoss(margin=margin, distance_function=distance_metric)
    logger.info(f"Loss function is created with the margin {margin} and distance function {distance_metric}.")
    record_loss = []
    record = 0
    record_valid_loss = []
    # Train
    model.train()
    for epoch in range(epochs):
        for idx, (embeddings, positive, negative) in enumerate(train_dataloader):
            # Zero the gradients
            optimizer.zero_grad()
            loss = loss_fn(embeddings, positive, negative)
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
            model.eval()
            if save_each_epoch:
                save_file = save_file.replace('.pth', f'_epoch_{epoch + 1}.pth')
            torch.save(model.state_dict(), save_file)
            logger.info(f"Model is saved to {save_file}.")
            save_file = save_file.replace(f'_epoch_{epoch + 1}.pth', '.pth')
            model.train()
        # Evaluate
        record /= len(train_dataloader)
        record_loss.append(record)
        wandb.log({"Training Loss": record})
        record = 0
        if validation_dataset is None:
            logger.warning("No validation dataset is provided. Skip validation.")
            continue # Skip validation if no validation dataset is provided
        model.eval()
        valid_dataloader = DataLoader(validation_dataset,
                                collate_fn=collate_fn,
                                batch_sampler=BalancedBatchSampler(validation_dataset, n_classes=n_classes, n_samples=n_samples),
                                **dataloader_kwargs)
        with torch.no_grad():
            valid_loss = 0
            for idx, (embeddings, positive, negative) in enumerate(valid_dataloader):
                valid_loss += loss_fn(embeddings, positive, negative).item()
            valid_loss /= len(valid_dataloader)
            logger.info(f"Epoch {epoch + 1}/{epochs} -- validation loss: {valid_loss:.4f}")
            wandb.log({"Validation Loss": valid_loss})
            record_valid_loss.append(valid_loss)
    logger.info("Training finished.")
    return record_loss, record_valid_loss