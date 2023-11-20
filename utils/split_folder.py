import os
from shutil import copytree, rmtree
import numpy as np
from tqdm import tqdm

import logging

def split_folder(dir: str, new_dir: str, ratio: float = 0.8):
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