import torch.nn as nn
import torchvision.models as models

def get_resnet18(feature_extract=True, num_classes=5):
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    if feature_extract:
        for param in model.parameters():
            param.requires_grad = False
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model

def get_densenet121(feature_extract=True, num_classes=5, dropout=0.0):
    model = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
    if feature_extract:
        for param in model.parameters():
            param.requires_grad = False
    in_features = model.classifier.in_features
    if dropout and dropout > 0.0:
        model.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, num_classes)
        )
    else:
        model.classifier = nn.Linear(in_features, num_classes)
    return model