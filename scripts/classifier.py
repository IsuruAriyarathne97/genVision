import torch
import torch.nn as nn
import torchvision.models as models

class ImageClassifier(nn.Module):
    def __init__(self, num_classes, model_name='resnet50', pretrained=True):
        super(ImageClassifier, self).__init__()
        
        # Load pre-trained ResNet model
        if model_name == 'resnet50':
            self.base_model = models.resnet50(pretrained=pretrained)
            num_features = self.base_model.fc.in_features
            self.base_model.fc = nn.Sequential(
                nn.Linear(num_features, 512),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(512, num_classes)
            )
        
    def forward(self, x):
        return self.base_model(x)