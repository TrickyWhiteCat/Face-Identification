from torch.utils.data import Dataset
import torch
from facenet_pytorch import MTCNN
from torchvision import transforms, models, datasets
from PIL import Image
import random

class FaceNetDataset(datasets.ImageFolder):
    def __init__(self, root, transform=None, target_transform=None, samples_per_label=4, mtcnn=None, resize_size=(112, 112)):
        super(FaceNetDataset, self).__init__(root=root, transform=transform, target_transform=target_transform)
        self.samples_per_label = samples_per_label
        self.mtcnn = MTCNN(keep_all=True)
        self.resize_size = resize_size
        self.to_tensor = transforms.ToTensor()

    def _extract_face(self, image_path):
        img = Image.open(image_path).convert("RGB")
        boxes, _ = self.mtcnn.detect(img)

        if boxes is not None and len(boxes) > 0:
            box = boxes[0]
            face_region = img.crop(box.astype(int))
            face_region = transforms.Resize(self.resize_size)(face_region)
            return self.to_tensor(face_region)
        else:
            # Return a tensor with zeros if no face is detected
            return torch.zeros(3, *self.resize_size)

    def __getitem__(self, index):
        img_anchor_path, label = self.samples[index]

        # Get positive index
        same_label_indices = [idx for idx in range(len(self.samples)) if self.samples[idx][1] == label]
        positive_index = random.choice(same_label_indices)

        # Get negative index
        different_label_indices = [idx for idx in range(len(self.samples)) if self.samples[idx][1] != label]
        negative_index = random.choice(different_label_indices)

        img_positive_path, _ = self.samples[positive_index]
        img_negative_path, _ = self.samples[negative_index]

        # Extract faces and convert to tensors
        img_anchor = self._extract_face(img_anchor_path)
        img_positive = self._extract_face(img_positive_path)
        img_negative = self._extract_face(img_negative_path)

        return img_anchor, img_positive, img_negative, label