# Since the dataset can be very large, we split it into n parts and copy them to the destination folder.

import os
from distutils.dir_util import copy_tree
from multiprocessing import Process, cpu_count

from tqdm import tqdm


def wrapper(src_dirs, dst_dir):
    for src_dir in tqdm(src_dirs):
        copy_tree(src_dir, os.path.join(dst_dir, os.path.basename(src_dir)))

def split_and_copy(src_dir, dst_dir, n):
    # Create n folders
    for i in range(n):
        os.makedirs(os.path.join(dst_dir, str(i)), exist_ok=True)
    # Copy files
    sub_dirs = [os.path.join(src_dir, sub_dir) for sub_dir in os.listdir(src_dir)]
    # Split sub_dirs into n parts
    partlength = len(sub_dirs) // n
    sub_dirs = [sub_dirs[i*partlength:(i+1)*partlength] for i in range(n)]
    # Copy
    processes = []
    for i in range(n):
        processes.append(Process(target=wrapper, args=(sub_dirs[i], os.path.join(dst_dir, str(i)))))
    for process in processes:
        process.start()
    while any([process.is_alive() for process in processes]):
        pass
    for process in processes:
        process.join()

if __name__ == '__main__':
    split_and_copy(r"C:\Users\nmttu\OneDrive - Hanoi University of Science and Technology\FaceData\ms1m-retinaface-t1\imgs",
                   r"C:\Users\nmttu\OneDrive - Hanoi University of Science and Technology\FaceData\ms1m-retinaface-t1\imgs_split",
                   90)