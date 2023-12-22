import torch
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import os
import tqdm

CACHED_DATA_INFO_PATH = r'data_info.json'

def mean_std(data_path, cache_path: str = None, cache: bool = True):
    # Check if data_path in cache
    if cache_path is not None:
        cache_path = CACHED_DATA_INFO_PATH
    if os.path.exists(cache_path):
        import json
        with open(cache_path, 'r') as f:
            data_info = json.load(f)
        if data_path in data_info:
            return data_info[data_path]['mean'], data_info[data_path]['std']

    dataset = datasets.ImageFolder(data_path, transform = transforms.ToTensor())

    loader = DataLoader(dataset,
                        batch_size=10,
                        )

    mean = 0.0
    with torch.no_grad():
        for images, _ in tqdm(loader):
            batch_samples = images.size(0) 
            images = images.view(batch_samples, images.size(1), -1)
            mean += images.mean(2).sum(0)
        mean = mean / len(loader.dataset)

        var = 0.0
        pixel_count = 0
        for images, _ in tqdm(loader):
            batch_samples = images.size(0)
            images = images.view(batch_samples, images.size(1), -1)
            var += ((images - mean.unsqueeze(1))**2).sum([0,2])
            pixel_count += images.nelement() / images.size(1)
        std = torch.sqrt(var / pixel_count)
    if cache:
        data_info[data_path] = {'mean': mean.tolist(), 'std': std.tolist()}
        with open(cache_path, 'w') as f:
            json.dump(data_info, f)
    return mean, std