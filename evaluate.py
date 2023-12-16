import logging

from tqdm import tqdm
from PIL import Image

logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger("Evaluation")
import os
import time

import numpy as np
import torch
from torchvision.transforms import v2 as transforms
import evaluate_utils as eutils
from arcface.head import ArcFaceEmbeddingHead

from torchvision import models

PAIR_PATH = r'data\pairs.txt'
DATA_DIR = r"data\lfw-deepfunneled-full\lfw-deepfunneled"
VALID_DATA_DIR = r'data\lfw-deepfunneled-split\valid'

device = 'cuda'
far = 0.01

# Model declaration and loading
model = models.mobilenet_v3_small()
model.classifier = ArcFaceEmbeddingHead(512, 576, last_batchnorm=True)
model.load_state_dict(torch.load(r"checkpoints\arcface\finetune_MobileNetV3_epoch_30.pth"))
model.eval().to(device)

transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Resize([224, 224], antialias=True),
                                transforms.Normalize(mean = [0, 0, 0], std=[0.2, 0.2, 0.2])])

embeddings = []
labels = []
stack = []
img_fullpaths = []
logger.info(f"Getting labels...")
list_labels = os.listdir(DATA_DIR)
for label in tqdm(list_labels):
    img_path = os.listdir(os.path.join(DATA_DIR, label))
    img_fullpaths += [os.path.join(DATA_DIR, label, img) for img in img_path]
    labels += [label] * len(img_path)
total_images = len(img_fullpaths)
labels = np.array(labels)

mini_batch_size = 32
logger.info("Start embedding...")
transform(Image.open(img_fullpaths[0])) # Initialize transformations
start = time.time()

for img_idx in tqdm(range(0, len(img_fullpaths))):
    img = transform(Image.open(img_fullpaths[img_idx]))
    stack.append(img)
    if len(stack) == mini_batch_size:
        embeddings.extend(eutils.batch_and_forward(model, stack, device).detach().cpu())
        stack = []
exec_time = time.time() - start
logger.info(f"Total execution time: {exec_time:.3f}s, approx. {exec_time/total_images:.5f}s per image or {total_images/exec_time:.3f} FPS.")
if len(stack) > 0:
    embeddings.extend(eutils.batch_and_forward(model, stack, device).detach().cpu())
    stack = []
embeddings = np.array(embeddings)
assert (len(embeddings) == len(labels)) and (len(embeddings) == total_images), f"Number of embeddings and labels must be equal to the number of images. Got {len(embeddings)} embeddings and {len(labels)} labels with {total_images} images."

embeddings_dict = dict(zip(img_fullpaths, embeddings))

pairs = eutils.read_pairs(PAIR_PATH)
path_list, issame_list = eutils.get_paths(DATA_DIR, pairs)
embeddings = np.array([embeddings_dict[path] for path in path_list])

logger.info(f"{total_images} embeddings and labels are loaded.")

print("Euclidean distance evaluation.")
tpr, fpr, accuracy, val, val_std, far, fp, fn = eutils.evaluate(embeddings, issame_list, far_target=far, distance_metric=0)
print(f'Accuracy: {np.mean(accuracy):.5f}+-{np.std(accuracy):.5f}')
print(f'Validation rate: {np.mean(val):.5f}+-{np.std(val)} @ FAR={far:.5f}')

print("Cosine similarity evaluation.")
tpr, fpr, accuracy, val, val_std, far, fp, fn = eutils.evaluate(embeddings, issame_list, far_target=far, distance_metric=1)
print(f'Accuracy: {np.mean(accuracy):.5f}+-{np.std(accuracy):.5f}')
print(f'Validation rate: {np.mean(val):.5f}+-{np.std(val)} @ FAR={far:.5f}')

'''
# Calculate distance


logger.info("Calculating threshold.")
threshold = threshold_from_far(far, model, embeddings, labels, device=device)
logger.info(f"Threshold: {threshold:.3f} is calculated from FAR: {far:.3f}.")
logger.info(f"Calculating confusion matrix.")

confusion_matrix = np.zeros((2, 2))
for idx1 in range(total_images):
    for idx2 in range(idx1 + 1, total_images):
        dist = np.linalg.norm(embeddings[idx1] - embeddings[idx2])
        pred = 1 if dist <= threshold else 0
        actual = 1 if labels[idx1] == labels[idx2] else 0
        confusion_matrix[actual, pred] += 1

val, far = calc_val_far(confusion_matrix)
print(pd.DataFrame(confusion_matrix, index=['actual 0', 'actual 1'], columns=['pred 0', 'pred 1']))
print(f'Validation rate: {val:.3f}\nFalse acceptance rate: {far:.3f}')
'''