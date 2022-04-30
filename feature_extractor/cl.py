import torch.nn as nn

class IClassifier(nn.Module):
    def __init__(self, feature_extractor, feature_size, output_class):
        super(IClassifier, self).__init__()
        # ResNet
        self.feature_extractor = feature_extractor
        # feature size (512) to output class (3)    
        self.fc = nn.Linear(feature_size, output_class)
        
        
    def forward(self, x):
        # resnet18/resnet50
        feats = self.feature_extractor(x) # N x K
        # c = self.fc(feats.view(feats.shape[0], -1)) # N x C
        return feats.view(feats.shape[0], -1)