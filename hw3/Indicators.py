import torch.nn as nn
import torch


class Indicators(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels, 6, 5),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(6, 16, 5),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(16 * 253 * 253, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, out_channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        features = self.feature_extractor(x)
        features = features.view(features.size(0), -1)
        class_scores = self.classifier(features)
        class_scores = class_scores.view(-1, 1, 10)
        #print(f"class_scores:{class_scores}")
        return class_scores


# N*C
def indicator_loss(scores, actual_indicators):
    #print(f"scores shape:{scores.shape}")
    #print(f"scores type:{scores.dtype}")
    #print(f"actual_indicators shape:{actual_indicators.shape}")
    #print(f"actual_indicators type:{actual_indicators.dtype}")
    loss = abs(actual_indicators - scores)
    #print(f"loss:{loss}")
    return loss.sum()

