import cv2
import torch
import torch.nn as nn
from torchvision import models, transforms
import numpy as np


class FeatureExtractor:
    def __init__(self):
        # Use MobileNet V2 as the architecture and ImageNet for the weight
        self.model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
        # Remove the last layer (output layer)
        self.model.classifier = nn.Sequential(*list(self.model.classifier.children())[:-1])
        self.model.eval()
        # Define image transforms
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def extract(self, img): 
        # Convert the image color space and resize image
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224, 224))
        # Apply image transforms
        img = self.transform(img)
        # Add batch dimension
        img = img.unsqueeze(0)
        # Extract Features, using gpu to boost up the performance
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        net = self.model.to(device)
        img = img.to(device)
        with torch.no_grad():
            feature = net(img).cpu().numpy().flatten()
        img = img.detach()
        torch.cuda.empty_cache()
        return feature / np.linalg.norm(feature) # normalize the feature vector
