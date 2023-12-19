import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ConvBlock, self).__init__()
        
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU()
        
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x

class InceptionF5(nn.Module):
    """
        From the paper, figure 5 inception module.
    """
    def __init__(self, in_channels):
        super(InceptionF5, self).__init__()
        
        self.branch1 = nn.Sequential(
            ConvBlock(in_channels, 64, kernel_size=1, stride=1, padding=0),
            ConvBlock(64, 96, kernel_size=3, stride=1, padding=1),
            ConvBlock(96, 96, kernel_size=3, stride=1, padding=1)
        )
        
        self.branch2 = nn.Sequential(
            ConvBlock(in_channels, 48, kernel_size=1, stride=1, padding=0),
            ConvBlock(48, 64, kernel_size=3, stride=1, padding=1)
        )
        
        self.branch3 = nn.Sequential(
            nn.MaxPool2d(3, stride=1, padding=1),
            ConvBlock(in_channels, 64, kernel_size=1, stride=1, padding=0)
        )
        
        self.branch4 = nn.Sequential(
            ConvBlock(in_channels, 64, kernel_size=1, stride=1, padding=0)
        )
        
    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)
        
        return torch.cat([branch1, branch2, branch3, branch4], 1)
    
class InceptionF6(nn.Module):
    """
        From the paper, figure 6 inception module.
    """
    def __init__(self, in_channels, f_7x7):
        super(InceptionF6, self).__init__()
        
        self.branch1 = nn.Sequential(
            ConvBlock(in_channels, f_7x7, kernel_size=1, stride=1, padding=0),
            ConvBlock(f_7x7, f_7x7, kernel_size=(1,7), stride=1, padding=(0,3)),
            ConvBlock(f_7x7, f_7x7, kernel_size=(7,1), stride=1, padding=(3,0)),
            ConvBlock(f_7x7, f_7x7, kernel_size=(1,7), stride=1, padding=(0,3)),
            ConvBlock(f_7x7, 192, kernel_size=(7,1), stride=1, padding=(3,0))
        )
        
        self.branch2 = nn.Sequential(
            ConvBlock(in_channels, f_7x7, kernel_size=1, stride=1, padding=0),
            ConvBlock(f_7x7, f_7x7, kernel_size=(1,7), stride=1, padding=(0,3)),
            ConvBlock(f_7x7, 192, kernel_size=(7,1), stride=1, padding=(3,0))
        )
        
        self.branch3 = nn.Sequential(
            nn.MaxPool2d(3, stride=1, padding=1),
            ConvBlock(in_channels, 192, kernel_size=1, stride=1, padding=0)
        )
        
        self.branch4 = nn.Sequential(
            ConvBlock(in_channels, 192, kernel_size=1, stride=1, padding=0)
        )
        
    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)
        
        return torch.cat([branch1, branch2, branch3, branch4], 1)

class InceptionF7(nn.Module):
    """
        From the paper, figure 7 inception module.
    """
    def __init__(self, in_channels):
        super(InceptionF7, self).__init__()
        
        self.branch1 = nn.Sequential(
            ConvBlock(in_channels, 448, kernel_size=1, stride=1, padding=0),
            ConvBlock(448, 384, kernel_size=(3,3), stride=1, padding=1)
        )
        self.branch1_top = ConvBlock(384, 384, kernel_size=(1,3), stride=1, padding=(0,1))
        self.branch1_bot = ConvBlock(384, 384, kernel_size=(3,1), stride=1, padding=(1,0))
        
        
        self.branch2 = ConvBlock(in_channels, 384, kernel_size=1, stride=1, padding=0)
        self.branch2_top = ConvBlock(384, 384, kernel_size=(1,3), stride=1, padding=(0,1))
        self.branch2_bot = ConvBlock(384, 384, kernel_size=(3,1), stride=1, padding=(1,0))
        
        self.branch3 = nn.Sequential(
            nn.MaxPool2d(3, stride=1, padding=1),
            ConvBlock(in_channels, 192, kernel_size=1, stride=1, padding=0)
        )
        
        self.branch4 = nn.Sequential(
            ConvBlock(in_channels, 320, kernel_size=1, stride=1, padding=0)
        )
        
    def forward(self, x):
        branch1 = self.branch1(x)
        branch1 = torch.cat([self.branch1_top(branch1), self.branch1_bot(branch1)], 1)
        
        branch2 = self.branch2(x)
        branch2 = torch.cat([self.branch2_top(branch2), self.branch2_bot(branch2)], 1)
        
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)
        
        return torch.cat([branch1, branch2, branch3, branch4], 1)

class InceptionRed(nn.Module):
    """
        From the paper, figure 10 improved pooling operation.
    """
    def __init__(self, in_channels, f_3x3_r, add_ch=0):
        super(InceptionRed, self).__init__()
        
        self.branch1 = nn.Sequential(
            ConvBlock(in_channels, f_3x3_r, kernel_size=1, stride=1, padding=0),
            ConvBlock(f_3x3_r, 178 + add_ch, kernel_size=3, stride=1, padding=1),
            ConvBlock(178 + add_ch, 178 + add_ch, kernel_size=3, stride=2, padding=0)
        )
        
        self.branch2 = nn.Sequential(
            ConvBlock(in_channels, f_3x3_r, kernel_size=1, stride=1, padding=0),
            ConvBlock(f_3x3_r, 302 + add_ch, kernel_size=3, stride=2, padding=0)
        )
        
        self.branch3 = nn.Sequential(
            nn.MaxPool2d(3, stride=2, padding=0)
        )
        
    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        
        return torch.cat([branch1, branch2, branch3], 1)
    
class InceptionAux(nn.Module):
    """
        From the paper, auxilary classifier
    """
    def __init__(self, in_channels, num_classes):
        super(InceptionAux, self).__init__()
        
        self.pool = nn.AdaptiveAvgPool2d((4,4))
        self.conv = nn.Conv2d(in_channels, 128, kernel_size=1, stride=1, padding=0)
        self.act = nn.ReLU()
        self.fc1 = nn.Linear(2048, 1024)
        self.dropout = nn.Dropout(0.7)
        self.fc2 = nn.Linear(1024, num_classes)
    
    def forward(self, x):
        x = self.pool(x)
        
        x = self.conv(x)
        x = self.act(x)
    
        x = torch.flatten(x, 1)
        
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        
        x = self.fc2(x)
        
        return x

class InceptionV2(nn.Module):
    """
    InceptionV2: An implementation of the Inception-v2 (GoogLeNet) architecture.

    This model is designed for image classification tasks and consists of various
    Inception modules, as described in the paper "Batch Normalization: Accelerating
    Deep Network Training by Reducing Internal Covariate Shift."

    Args:
        num_classes (int): Number of output classes for classification. Default is 10.

    Attributes:
        resize (nn.Upsample): Upsampling layer to standardize input image size.
        conv1-conv6 (ConvBlock): Convolutional blocks used in the initial layers.
        inception3a, inception3b, inception3c (InceptionF5): Inception modules for
            the third stage.
        inceptionRed1 (InceptionRed): Reduction module after the third stage.
        inception4a, inception4b, inception4c, inception4d, inception4e (InceptionF6):
            Inception modules for the fourth stage.
        inceptionRed2 (InceptionRed): Reduction module after the fourth stage.
        aux (InceptionAux): Auxiliary classifier for intermediate supervision.
        inception5a, inception5b (InceptionF7): Inception modules for the fifth stage.
        pool6 (nn.AdaptiveAvgPool2d): Adaptive average pooling layer.
        dropout (nn.Dropout): Dropout layer.
        fc (nn.Linear): Fully connected layer for classification.
        embedding_fc (nn.Linear): Fully connected layer for producing embeddings.

    Methods:
        forward(x): Forward pass through the network.

    Example:
        # Create an InceptionV2 model with 10 output classes
        model = InceptionV2(num_classes=10)
        # Forward pass with input tensor x
        output, aux_output, embedding = model(x)
    """
    
    def __init__(self, num_classes = 10):
        super(InceptionV2, self).__init__()
        
        # Thêm lớp chuyển đổi kích thước cho ảnh đầu vào
        self.resize = nn.Upsample(size=(112, 112), mode='bilinear', align_corners=False)
        
        self.conv1 = ConvBlock(3, 32, kernel_size=3, stride=2, padding=0)
        self.conv2 = ConvBlock(32, 32, kernel_size=3, stride=1, padding=0)
        self.conv3 = ConvBlock(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(3, stride=2, padding=0)
        self.conv4 = ConvBlock(64, 80, kernel_size=3, stride=1, padding=0)
        self.conv5 = ConvBlock(80, 192, kernel_size=3, stride=2, padding=0)
        self.conv6 = ConvBlock(192, 288, kernel_size=3, stride=1, padding=1)
        
        self.inception3a = InceptionF5(288)
        self.inception3b = InceptionF5(288)
        self.inception3c = InceptionF5(288)
        
        self.inceptionRed1 = InceptionRed(288,f_3x3_r=64, add_ch=0)
        
        self.inception4a = InceptionF6(768, f_7x7=128)
        self.inception4b = InceptionF6(768, f_7x7=160)
        self.inception4c = InceptionF6(768, f_7x7=160)
        self.inception4d = InceptionF6(768, f_7x7=160)
        self.inception4e = InceptionF6(768, f_7x7=192)
        
        self.inceptionRed2 = InceptionRed(768,f_3x3_r=192, add_ch=16)
        
        self.aux = InceptionAux(768, num_classes) 
        
        self.inception5a = InceptionF7(1280)
        self.inception5b = InceptionF7(2048)
        
        self.pool6 = nn.AdaptiveAvgPool2d((1,1))
        self.dropout = nn.Dropout(0.4)
        self.fc = nn.Linear(128, num_classes)
        
        # Add a fully connected layer for the final embedding
        self.embedding_fc = nn.Linear(2048, 128)
    
    def forward(self, x):
        x = self.resize(x)
        
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.pool1(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        
        x = self.inception3a(x)
        x = self.inception3b(x)
        x = self.inception3c(x)

        x = self.inceptionRed1(x)
        
        x = self.inception4a(x)
        x = self.inception4b(x)
        x = self.inception4c(x)
        x = self.inception4d(x)
        x = self.inception4e(x)
        
        aux = self.aux(x)
        
        x = self.inceptionRed2(x)    
        x = self.inception5a(x)
        embedding = self.inception5b(x)  # Output from inception5b

        # Global Average Pooling (GAP)
        embedding = F.adaptive_avg_pool2d(embedding, (1, 1))

        # Flatten and pass through the fully connected layer
        embedding = torch.flatten(embedding, 1)
        embedding = self.embedding_fc(embedding)

        x = self.fc(embedding) 

        return x, aux, embedding.squeeze()  