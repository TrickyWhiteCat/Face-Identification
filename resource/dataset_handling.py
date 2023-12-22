import random
from facenet_pytorch import MTCNN
from torchvision import transforms,datasets
import torch
from PIL import Image
class MyDataset(datasets.ImageFolder):
    def __init__(self, root, transform=None, resize_size=(256, 256)):
        super(MyDataset, self).__init__(root=root,transform=transform)
        self.mtcnn = MTCNN(keep_all=True)
        self.resize_size = resize_size
        self.to_tensor = transforms.ToTensor()

    def face_transform(self, image_path):
        img = Image.open(image_path).convert("RGB")
        boxes, _ = self.mtcnn.detect(img)

        if boxes is not None and len(boxes) > 0:
            box = boxes[0]
            face_region = img.crop(box.astype(int))
            return self.transform(face_region)
        else:
            # Return a tensor with zeros if no face is detected
            return torch.zeros(3, *self.resize_size)

    def __getitem__(self, index):
        anchor_path, label = self.samples[index]

        # Get positive index
        same_label_indices = [idx for idx in range(len(self.samples)) if self.samples[idx][1] == label]
        positive_index = random.choice(same_label_indices)

        # Get negative index
        different_label_indices = [idx for idx in range(len(self.samples)) if self.samples[idx][1] != label]
        negative_index = random.choice(different_label_indices)

        positive_path, _ = self.samples[positive_index]
        negative_path, _ = self.samples[negative_index]
        # Extract faces and convert to tensors
        anchor = self.face_transform(anchor_path)
        positive = self.face_transform(positive_path)
        negative = self.face_transform(negative_path)

        return anchor, positive, negative