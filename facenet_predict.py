
import os
import random

import torch
from torch import nn, threshold
from torchvision import models
from torchvision.io import read_image, write_jpeg

import numpy as np

from matplotlib import pyplot as plt

from tqdm import tqdm


save_dir = 'checkpoints'
data_dir = f'data/lfw-deepfunneled-split/valid'


model = models.mobilenet_v3_small(weights = models.MobileNet_V3_Small_Weights.DEFAULT)
model.classifier = torch.nn.Sequential(torch.nn.AdaptiveAvgPool2d((1, 1)), torch.nn.Linear(576, 128))
transform = models.MobileNet_V3_Small_Weights.DEFAULT.transforms(antialias=True)

model.load_state_dict(torch.load(os.path.join(save_dir, r'hard_MobileNetV3_(0.001, 1e-05)_euclidean_epoch_100.pth')))
model.eval()


device = 'cpu'
model.to(device)

img_per_label = 3
num_labels = 4

list_labels = os.listdir(data_dir)
diff_labels = random.sample(list_labels, num_labels)

'''
# Get n images from each label
images = []
for label in diff_labels:
    list_images = os.listdir(os.path.join(data_dir, label))
    [images.append(os.path.join(data_dir, label, img_path)) for img_path in random.sample(list_images, img_per_label)]

# Read all images and convert to tensor
batched = torch.stack([transform(read_image(image)) for image in images])
# Get embeddings
embeddings = torch.nn.functional.normalize(model(batched.to(device)), p=2, dim=1)
# Calculate distance
dist = torch.cdist(embeddings, embeddings)
np.set_printoptions(2)
print(dist.detach().cpu().numpy())
plt.imshow(dist.detach().cpu().numpy())
plt.show()
'''

# Embed all images and save to a file
EMBEDDING_FILE = 'embeddings.txt'
LABEL_FILE = 'labels.txt'
'''
# Reset file
open(EMBEDDING_FILE, 'w').close()
open(LABEL_FILE, 'w').close()


mini_batch_size = 16
for label in list_labels:
    list_images = os.listdir(os.path.join(data_dir, label))
    for i in range(0, len(list_images), mini_batch_size):
        batched = torch.stack([transform(read_image(os.path.join(data_dir, label, img_path))) for img_path in list_images[i:i+mini_batch_size]])
        embeddings = torch.nn.functional.normalize(model(batched.to(device)), p=2, dim=1)
        with open(EMBEDDING_FILE, 'a') as f:
            for embedding in embeddings:
                f.write(' '.join([str(x) for x in embedding.detach().cpu().numpy()]) + '\n')
        with open(LABEL_FILE, 'a') as f:
            for _ in range(len(list_images[i:i+mini_batch_size])):
                f.write(label + '\n')
'''
embeddings = np.loadtxt(EMBEDDING_FILE)
labels = np.loadtxt(LABEL_FILE, dtype=str)
label_idx = {label: idx for idx, label in enumerate(np.unique(labels))}
labels_idx = np.array([label_idx[label] for label in labels]).reshape(-1, 1)

consfusion_matrix = np.zeros((2, 2))
threshold = 0.8
for idx1 in range(len(embeddings)):
    for idx2 in range(idx1+1, len(embeddings)):
        if np.linalg.norm(embeddings[idx1] - embeddings[idx2]) < threshold:
            pred = 1
        else:
            pred = 0
        if labels_idx[idx1] == labels_idx[idx2]:
            actual = 1
        else:
            actual = 0
        consfusion_matrix[actual, pred] += 1

print('*'*os.get_terminal_size().columns)
import pandas as pd
print(pd.DataFrame(consfusion_matrix, index=['Actual Negative', 'Actual Positive'], columns=['Predicted Negative', 'Predicted Positive']))
def metrics(confusion_matrix):
    TN, FP, FN, TP = confusion_matrix.ravel()
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1_score = 2 * precision * recall / (precision + recall)
    return accuracy, precision, recall, f1_score

acc, prec, rec, f1 = metrics(consfusion_matrix)
print('*'*os.get_terminal_size().columns)
print(f'Accuracy: {acc:.3%}\nPrecision: {prec:.3%}\nRecall: {rec:.3%}\nF1-score: {f1:.3%}')
print('*'*os.get_terminal_size().columns)