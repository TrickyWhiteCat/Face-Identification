'''
FaceNet is a face recognition system developed in 2015 that achieved then state-of-the-art results on a range of face recognition benchmark datasets described in the paper FaceNet: A Unified Embedding for Face Recognition and Clustering (https://arxiv.org/pdf/1503.03832.pdf). The model proposed is a deep convolutional network trained to directly optimize the embedding itself, rather than an intermediate bottleneck layer as in previous deep learning approaches. The FaceNet model can be used to extract high-quality features from faces, called face embeddings, that can then be used to train a face identification system.
'''

import logging
import os
from typing import Callable

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

from facenet.sampler import BalancedBatchSampler


class FaceNet(nn.Module):
    def __init__(self, extractor: nn.Module, data_dir: str, distance_metric: str|Callable = 'euclidean',embedding_size: int = 128, margin: float = 1.0, transform = None, device: str = 'cpu', *, skip_data_check: bool = False):
        '''Initializes the FaceNet wrapper model which enables training and embedding described in the paper FaceNet: A Unified Embedding for Face Recognition and Clustering (https://arxiv.org/pdf/1503.03832.pdf).
        Parameters:
            extractor (nn.Module): The feature extractor model.
            distance_metric (str, Callable, optional): The distance metric to use when comparing face embeddings. If Callable, must take 2 tensors and an optional parameter `dim` to specify the dimension to perform calculation. Defaults to 'euclidean'.
            embedding_size (int, optional): The size of the embedding vector. Defaults to 128.
            threshold (float, optional): The threshold to use when comparing face embeddings. Defaults to 1.0.
            device (str, optional): The device to use for the model. Defaults to 'cpu'.
        '''
        logger = logging.getLogger("FaceNetInitializer")
        logger.info("Initializing parent class.")
        super().__init__()

        self.extractor = extractor
        logger.info(f"Extractor is set to {extractor}.")
        self.DATA_DIR = self.__check_dir(data_dir) if not skip_data_check else data_dir
        logger.info(f"Data directory is set to {data_dir}.")
        # Optional parameters
        self.distance_metric = self.__distance_metric_factory(distance_metric)
        logger.info(f"Distance metric is set to {distance_metric}.")
        self.embedding_size = embedding_size if embedding_size > 0 else 128
        logger.info(f"Embedding size is set to {embedding_size}.")
        self.margin = margin
        logger.info(f"Margin is set to {margin}.")
        self.transform = transform if transform is not None else transforms.ToTensor()
        logger.info(f"Transform is set to {self.transform}.")

        self.linear_head = nn.LazyLinear(self.embedding_size)
        self.dataset = datasets.ImageFolder(self.DATA_DIR, transform=self.transform)
        logger.info(f"Dataset is created.")

        if device == 'cuda':
            if torch.cuda.is_available():
                self.device = 'cuda'
            else:
                logger.warning("CUDA is not available, using CPU instead.")
                self.device = 'cpu'
        else:
            self.device = 'cpu'
        self.to(self.device)

        logger.info(f"Device is set to {self.device}.")

    
    def __check_dir(self, data_dir: str):
        '''Checks if the given directory exists and follows the expected structure: <data_dir>/<identity_name>/<image_name>.<extension>.
        Parameters:
            data_dir (str): The directory to check.
        Returns:
            str: The data directory.
        '''
        logger = logging.getLogger("DirectoryChecker")
        # Check path exists
        if not os.path.isdir(data_dir):
            raise ValueError('Directory does not exist: {}'.format(data_dir))
        logger.info(f"Directory {data_dir} exists.")
        # Check structure
        for identity in os.listdir(data_dir):
            if not os.path.isdir(os.path.join(data_dir, identity)):
                err_msg = f"Expected directory structure: <data_dir>/<identity_name>/<image_name>.<extension>, but found {os.path.join(data_dir, identity)}"
                logger.error(err_msg)
                raise ValueError(err_msg)
            for image in os.listdir(os.path.join(data_dir, identity)):
                if not os.path.isfile(os.path.join(data_dir, identity, image)):
                    err_msg = f"Expected directory structure: <data_dir>/<identity_name>/<image_name>.<extension>, but found {os.path.join(data_dir, identity, image)}"
                    logger.error(err_msg)
                    raise ValueError(err_msg)
        logger.info(f"Directory structure is correct.")
        return data_dir

    def __distance_metric_factory(self, distance_metric: str|Callable):
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

    def forward(self, x: torch.Tensor):
        '''Forward pass of the model.
        Parameters:
            x (torch.Tensor): The input tensor.
        Returns:
            torch.Tensor: The output tensor.
        '''
        logger = logging.getLogger("FaceNetForward")
        logger.info(f"Forward pass started with the input shape {x.shape}.")
        logger.info(f"Input is moved to {self.device}.")
        # Extract features
        features = self.extractor(x)
        logger.info(f"Features are extracted with the shape {features.shape}.")
        # Flatten features
        features = torch.flatten(features, start_dim=1)
        logger.info(f"Features are flattened with the shape {features.shape}.")
        # Linear head
        embeddings = self.linear_head(features)
        logger.info(f"Embeddings are calculated with the shape {embeddings.shape}.")
        return embeddings
    
    def __select_triplets(self, mini_batch: torch.Tensor, semihard_negative:bool = True):
        '''Selects triplets for training based on the method described in the FaceNet paper. Given a mini-batch, first the function will embed the images using the model, then it will select triplets based on the distance between the embeddings. For each sample in the mini-batch, first it is used as an anchor and then a hard positive and a semi-hard negative sample is selected. A hard positive is the sample with the same label as the anchor that is the furthest away from the anchor. A semi-hard negative is the closest sample with a different label than the anchor but still further away than the positive sample within the margin.
        Parameters:
            mini_batch (torch.Tensor): The mini-batch of images with the shape (N, C, H, W)
            labels (torch.Tensor): The labels of the mini-batch.
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

        # Move mini-batch to device
        images = images.to(self.device)
        labels = labels.to(self.device)
        logger.info(f"Mini-batch is moved to {self.device}.")
        # Embed images
        embeddings = torch.nn.functional.normalize(self.forward(images), 2, 1) # (N, embedding_size)
        logger.info(f"Batch is embedded into {embeddings.shape}.")
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
            distance = self.distance_metric(embeddings.unsqueeze(1), embeddings.unsqueeze(0), dim=-1)
        except RuntimeError as e:
            logger.error(f"Vectorized implementation failed: {e}")
            # Clear CUDA cache
            torch.cuda.empty_cache()
            # For-loop implementation
            logger.info(f"Trying for-loop implementation.")
            distance = torch.zeros((embeddings.shape[0], embeddings.shape[0]), device=self.device)
            for i in range(embeddings.shape[0]):
                for j in range(embeddings.shape[0]):
                    distance[i, j] = self.distance_metric(embeddings[i], embeddings[j])

        # Create mask for positive and negative samples
        positive_mask = labels.unsqueeze(1) == labels.unsqueeze(0) # is_positive[i, j] = 1 if i and j have the same label, 0 otherwise
        negative_mask = ~positive_mask

        # Select hard positive samples
        # For each anchor, select the furthest positive sample
        positive_distance, hard_positive_idx = torch.max(distance * positive_mask.float(), dim=1)
        positive = torch.gather(embeddings, 0, hard_positive_idx.unsqueeze(1).repeat(1, embeddings.shape[1]))
        further_than_positive = distance > positive_distance.unsqueeze(1)
        logger.info(f"Hard positive samples are selected with the shape {positive.shape}.")

        if semihard_negative:
            # Select semi-hard negative samples
            neg_further_than_positive = further_than_positive * negative_mask
            # For each anchor, select the closest negative sample that is further away than the positive sample
            _, negative_idx = torch.min(distance * neg_further_than_positive.float() + (~neg_further_than_positive).long() * 10e6, dim=1)
        else:
            # Select hard negative samples
            # For each anchor, select the closest negative sample
            _, negative_idx = torch.min(distance * negative_mask.float() + positive_mask.long() * 10e6 , dim=1)
        negative = torch.gather(embeddings, 0, negative_idx.unsqueeze(1).repeat(1, embeddings.shape[1]))
        logger.info(f"Semi-hard negative samples are selected with the shape {negative.shape}.")
        return embeddings, positive, negative
    
    def autotrain(self, learning_rate: float = None, n_classes = 8, n_samples = 4, epochs: int = 10, semihard_negative:bool = True, save_file:str = None, verbose = True, **dataloader_kwargs):
        '''
        Train the model with the given learning rate.
        Parameters:
            learning_rate (float, optional): The learning rate to use for training. Defaults to None.
        '''
        logger = logging.getLogger("FaceNetTrainer")
        logger.info("Training started.")
        # Create data loader
        if semihard_negative:
            collate_fn = self.__select_triplets
        else:
            collate_fn = lambda x: self.__select_triplets(x, semihard_negative=False)
        data_loader = DataLoader(self.dataset,
                                 collate_fn=collate_fn,
                                 batch_sampler=BalancedBatchSampler(self.dataset, n_classes=n_classes, n_samples=n_samples),
                                 **dataloader_kwargs)
        logger.info(f"Data loader is created with {n_classes} classes, {n_samples} per classes and epochs {epochs}.")
        # Create optimizer
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        logger.info(f"Optimizer is created with the learning rate {learning_rate}.")
        # Create loss function
        loss_fn = nn.TripletMarginWithDistanceLoss(margin=self.margin, distance_function=self.distance_metric)
        logger.info(f"Loss function is created with the margin {self.margin} and distance function {self.distance_metric}.")
        record_loss = []
        # Train
        self.train()
        for epoch in range(epochs):
            for idx, (embeddings, hard_positive, semi_hard_negative) in enumerate(data_loader):
                # Zero the gradients
                optimizer.zero_grad()
                # Forward pass
                loss = loss_fn(embeddings, hard_positive, semi_hard_negative)
                # Backward pass
                loss.backward()
                # Update parameters
                optimizer.step()

                if verbose:
                    print(f"Epoch {epoch + 1}/{epochs} -- batch {idx + 1}/{len(data_loader)} -- loss: {loss.item():.4f}", end  = '\r')
                record_loss.append(loss.item())
            # Save model
            if save_file is not None:
                torch.save(self.state_dict(), save_file)
                logger.info(f"Model is saved to {save_file}.")
        logger.info("Training finished.")
        return record_loss