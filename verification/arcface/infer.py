from pyexpat import model
from head import ArcFaceEmbeddingHead
from PIL import Image
import torch
import torchvision.transforms.v2 as transforms
from torchvision import models

CHECKPOINT_PATH = r""
IMG1_PATH = r""
IMG2_PATH = r""
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
THRESHOLD = 1


model = models.mobilenet_v3_small()
model.classifier = ArcFaceEmbeddingHead(128, 576)
model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE)['model_state_dict'])
model.eval()
model.to(DEVICE)

transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Resize([224, 224], antialias=True),
                                transforms.Normalize(mean = [0.5313, 0.4263, 0.3748], std=[0.2873, 0.2552, 0.2492])])

img1 = transform(Image.open(IMG1_PATH)).unsqueeze(0).to(DEVICE)
img2 = transform(Image.open(IMG2_PATH)).unsqueeze(0).to(DEVICE)
emb1 = model(img1)
emb2 = model(img2)
dist = torch.dist(emb1, emb2)
print("Distance:", dist)
print("Same person: ", dist < THRESHOLD)