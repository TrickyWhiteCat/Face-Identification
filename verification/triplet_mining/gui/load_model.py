import  torch.nn as nn
import torch.nn.functional as F
import torch
from PIL import Image    
from torchvision import transforms
import readFile
import cv2
from facenet_pytorch import MTCNN
class VGG16_NET(nn.Module):
    def __init__(self):
        super(VGG16_NET, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)

        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)

        self.conv5 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.conv7 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)

        self.conv8 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1)
        self.conv9 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.conv10 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)

        self.conv11 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.conv12 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.conv13 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.fc14 = nn.Linear(25088, 4096)
        self.fc15 = nn.Linear(4096, 4096)
        self.fc16 = nn.Linear(4096, 256)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.maxpool(x)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.maxpool(x)
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = F.relu(self.conv7(x))
        x = self.maxpool(x)
        x = F.relu(self.conv8(x))
        x = F.relu(self.conv9(x))
        x = F.relu(self.conv10(x))
        x = self.maxpool(x)
        x = F.relu(self.conv11(x))
        x = F.relu(self.conv12(x))
        x = F.relu(self.conv13(x))
        x = self.maxpool(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc14(x))
        x = F.dropout(x, 0.5)
        x = F.relu(self.fc15(x))
        x = F.dropout(x, 0.5)
        x = self.fc16(x)
        return x


def LoadModel(model_path,model):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the state_dict
    state_dict = torch.load(model_path, map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    # If the model was saved using DataParallel, strip the 'module.' prefix
    if 'module.' in list(state_dict.keys())[0]:
        state_dict = {key.replace('module.', ''): value for key, value in state_dict.items()}
    model.to(device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def calculateDistance(model, source_path, destination_frame):
    # Define MTCNN module
    mtcnn = MTCNN(keep_all=True, device='cuda' if torch.cuda.is_available() else 'cpu')

    aka = [source_path, destination_frame]
    list_embedding = []

    for i, input_data in enumerate(aka):
        if i == 0:  # Source image
            image = Image.open(input_data)
        else:  # Destination frame
            image = Image.fromarray(cv2.cvtColor(input_data, cv2.COLOR_BGR2RGB))

        # Detect faces using MTCNN
        boxes, probs = mtcnn.detect(image)
        if boxes is not None:
            # Crop face using the first detected face
            box = boxes[0]
            face = image.crop(box.astype(int))

            preprocess = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
            ])

            input_tensor = preprocess(face)
            input_tensor = input_tensor.unsqueeze(0)

            # Move to GPU if available
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            input_tensor = input_tensor.to(device)

            # Perform inference
            with torch.no_grad():
                output = model(input_tensor)

            list_embedding.append(output)
            
    # Normalize embeddings
    for i in range(len(list_embedding)):
        list_embedding[i] = torch.nn.functional.normalize(list_embedding[i], p=2, dim=1)

    distance = round(torch.nn.functional.pairwise_distance(list_embedding[0], list_embedding[1]).item(), 3)
    return distance

def getSamllestDistance(model,dest,threshold):
    folder_path = r"database"
    status=float('inf')
    image_names=readFile.get_person_folders(folder_path)
    for img,name in image_names:
        curr=calculateDistance(model,img,dest)
        if curr<=status:
            status=curr
            label=name
        print(name,status)
    if status<=threshold:
        return (status,label)
    else:
        return (-1,-1)


