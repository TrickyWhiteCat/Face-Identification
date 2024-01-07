import os
from shutil import copytree, rmtree, copyfile
import numpy as np
from tqdm import tqdm

import logging

def split_folder(dir: str, new_dir: str, ratio: float = 0.8, reuse=False):
    logger = logging.getLogger("FolderSplitter")
    if not os.path.isdir(dir):
        msg = f'{dir} is not a folder.'
        logger.error(msg)
        raise FileNotFoundError(msg)
    if not os.path.isdir(new_dir):
        os.makedirs(new_dir, exist_ok=True)
        logger.warn(f'{new_dir} does not exist, creating it.')
    train_dir = os.path.join(new_dir, 'train')
    valid_dir = os.path.join(new_dir, 'valid')
    if reuse:
        if os.path.isdir(train_dir) and os.path.isdir(valid_dir):
            logger.info(f'Reusing {train_dir} and {valid_dir}.')
            return train_dir, valid_dir
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(valid_dir, exist_ok=True)
    logger.info(f"Train folder: {train_dir} and Valid folder: {valid_dir} are created.")
    # Delete old folders
    logger.info(f'Deleting old folders in {train_dir} and {valid_dir}.')
    rmtree(train_dir)
    rmtree(valid_dir)
    folders = os.listdir(dir)
    logger.info(f'Found {len(folders)} folders in {dir}.')
    train_folders = np.random.choice(folders, size=int(len(folders) * ratio), replace=False).tolist()
    valid_folders = [folder for folder in folders if folder not in train_folders]
    logger.info(f'Copying {len(train_folders)} folders to {train_dir}')
    for folder in tqdm(train_folders):
        copytree(os.path.join(dir, folder), os.path.join(train_dir, folder))
    logger.info(f'Copying {len(valid_folders)} folders to {valid_dir}')
    for folder in tqdm(valid_folders):
        copytree(os.path.join(dir, folder), os.path.join(valid_dir, folder))
    return train_dir, valid_dir

def split_each_class(dir: str, new_dir: str, ratio: float = 0.8, reuse=False):
    '''Split data from `dir` to `new_dir` to train and validation folders. For each class, the data will be split with the given ratio.'''
    logger = logging.getLogger("PerClassSplitter")
    if not os.path.isdir(dir):
        msg = f'{dir} is not a folder.'
        logger.error(msg)
        raise FileNotFoundError(msg)
    if not os.path.isdir(new_dir):
        os.makedirs(new_dir, exist_ok=True)
        logger.warn(f'{new_dir} does not exist, creating it.')
    train_dir = os.path.join(new_dir, 'train')
    valid_dir = os.path.join(new_dir, 'valid')
    if reuse:
        if os.path.isdir(train_dir) and os.path.isdir(valid_dir):
            logger.info(f'Reusing {train_dir} and {valid_dir}.')
            return train_dir, valid_dir
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(valid_dir, exist_ok=True)
    logger.info(f"Train folder: {train_dir} and Valid folder: {valid_dir} are created.")
    # Delete old folders
    logger.info(f'Deleting old folders in {train_dir} and {valid_dir}.')
    rmtree(train_dir)
    rmtree(valid_dir)
    folders = os.listdir(dir)
    logger.info(f'Found {len(folders)} folders in {dir}.')
    for folder in tqdm(folders):
        folder_dir = os.path.join(dir, folder)
        files = os.listdir(folder_dir)
        np.random.shuffle(files)
        train_files = files[:int(len(files) * ratio)]
        valid_files = files[int(len(files) * ratio):]
        train_folder_dir = os.path.join(train_dir, folder)
        valid_folder_dir = os.path.join(valid_dir, folder)
        os.makedirs(train_folder_dir, exist_ok=True)
        os.makedirs(valid_folder_dir, exist_ok=True)
        for file in train_files:
            copyfile(os.path.join(folder_dir, file), os.path.join(train_folder_dir, file))
        for file in valid_files:
            copyfile(os.path.join(folder_dir, file), os.path.join(valid_folder_dir, file))
    return train_dir, valid_dir

def split(dir: str, new_dir: str, ratio: float = 0.8, type = 'sample', reuse=False):
    if type == 'sample':
        return split_each_class(dir, new_dir, ratio, reuse)
    if type == 'label':
        return split_folder(dir, new_dir, ratio, reuse)
    raise ValueError(f'Unknown type {type}.')