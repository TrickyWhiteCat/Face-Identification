# MobileNetV3 with ArcFace loss

### Train the model
1. Open [`main.py`](main.py), set `DATA_DIR` to the path of your dataset which should be organized so that it can be loaded by [`torchvision.datasets.ImageFolder`](https://pytorch.org/vision/stable/generated/torchvision.datasets.ImageFolder.html) object.
2. Also change other parameters in `main.py` if necessary.
3. Run `python main.py` to start training.

### Inference

1. Open [`infer.py`](infer.py) and set `CHECKPOINT_PATH` to the path of the checkpoint, `IMG1_PATH` and `IMG2_PATH` to the path of the images to be tested. You can also set the `THRESHOLD` to control the distance threshold of the verification.
2. Run `python infer.py` to start inference.

### Evaluation
1. Open [`evaluate.ipynb`](evaluate.ipynb) 
2. Head to the `Constants` section and set the `LFW_DIR` to the path of the LFW dataset and `PAIR_PATH` to the path of the `pairs.txt` file.
3. Copy this code and paste into the `Load the model` section
```python
from torchvision import models
from torchvision import transforms
from torch import nn

class ArcFaceEmbeddingHead(nn.Module):
    def __init__(self, embedding_size, in_features, dropout=0.2, last_batchnorm=True):
        super().__init__()
        self.batchnorm1 = nn.BatchNorm1d(in_features)
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(in_features, embedding_size, bias=True)
        self.features = nn.BatchNorm1d(embedding_size) if last_batchnorm else nn.Identity()
        nn.init.constant_(self.features.weight, 1.0)
        self.features.weight.requires_grad = False

    def forward(self, x):
        x = self.batchnorm1(x)
        x = self.dropout(x)
        x = self.fc(x)
        x = self.features(x)
        return nn.functional.normalize(x, dim=-1)

CHECKPOINT_PATH = <path to the checkpoint>
model = models.mobilenet_v3_small()
model.classifier = ArcFaceEmbeddingHead(128, 576)
model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE)['model_state_dict'])
model.eval()
model.to(DEVICE)

transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Resize([224, 224], antialias=True),
                                transforms.Normalize(mean = [0.5313, 0.4263, 0.3748], std=[0.2873, 0.2552, 0.2492])])
```
4. Run all cells to evaluate the model.