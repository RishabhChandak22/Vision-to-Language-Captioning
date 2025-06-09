import torch
import torch.nn as nn
import torchvision.models as models

class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad = False  


        self.resnet = nn.Sequential(*list(resnet.children())[:-1])  
        self.linear = nn.Linear(resnet.fc.in_features, embed_size)  
        self.bn = nn.BatchNorm1d(embed_size)

    def forward(self, images):
        with torch.no_grad():
            features = self.resnet(images).squeeze()  
        features = self.linear(features)             
        features = self.bn(features)
        return features

