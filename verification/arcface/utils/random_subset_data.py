# Random sampling data folders from a dataset and copy it to the destination folder.

import os
import random
from distutils.dir_util import copy_tree
from multiprocessing import Process, cpu_count

from tqdm import tqdm

def wrapper(src_dirs, dst_dir):
    for src_dir in tqdm(src_dirs):
        copy_tree(src_dir, os.path.join(dst_dir, os.path.basename(src_dir)))

def multicore_copy(src_dirs, dst_dir, n: cpu_count()):
    # Split src_dirs into n parts
    partlength = len(src_dirs) // n
    sub_dirs = [src_dirs[i*partlength:(i+1)*partlength] for i in range(n - 1)]
    sub_dirs.append(src_dirs[(n-1)*partlength:]) # The last part may have more elements than the others
    # Copy
    processes = []
    for i in range(n):
        processes.append(Process(target=wrapper, args=(sub_dirs[i], dst_dir)))
    for process in processes:
        process.start()
    while any([process.is_alive() for process in processes]):
        pass
    for process in processes:
        process.join()

def random_subset_data(src_dir, dst_dir, subset_size):
    # Randomly sample subset_size folders from src_dir
    subset = random.sample(os.listdir(src_dir), 
                           k = subset_size)
    subset = [os.path.join(src_dir, folder) for folder in subset]

    multicore_copy(subset, dst_dir, cpu_count())

if __name__ == '__main__':
    random_subset_data(r"C:\Users\nmttu\OneDrive - Hanoi University of Science and Technology\FaceData\ms1m-retinaface-t1\imgs",
                       r"C:\Users\nmttu\OneDrive - Hanoi University of Science and Technology\FaceData\ms1m-retinaface-t1\imgs_subset",
                       1000)