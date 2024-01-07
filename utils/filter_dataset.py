import os
import shutil
from tqdm import tqdm

def filter_dataset(data_dir: str, minimum_samples_per_label: int = 1):
    total = 0
    for label in tqdm(os.listdir(data_dir)):
        label_dir = os.path.join(data_dir, label)
        if len(os.listdir(label_dir)) < minimum_samples_per_label:
            shutil.rmtree(label_dir)
        else:
            print(f'Label {label} has {len(os.listdir(label_dir))} samples')
            total += len(os.listdir(label_dir))
    return total

print(filter_dataset('data/lfw-deepfunneled', 4))