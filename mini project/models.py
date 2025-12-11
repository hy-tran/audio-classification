import torch
import torch.nn as nn

# -----------------------------------------------------------------------------
# MODEL IMPLEMENTATION NOTES
#
# Goal:
#   Build a model that consumes the audio features produced by `LoadAudio`
#   and outputs logits over the audio scene classes.
#
# Features:
#   - You may pass *multiple* features from the DataLoader  
#     Keep their order consistent with the dataset.
#   - Example loader usage:
#         for (feat1, feat2, labels) in loader:
#             logits = model(feat1, feat2)
#
# 1D vs 2D:
#   - Starting with 1D raw waveforms is fine.
#   - Converting to 2D (Spectrograms / Mel-Spectrograms) is common and often
#     improves performance; design the network (e.g., CNNs) to match the feature
#     dimensionality.
#
# Evaluation requirements:
#   - Forward must accept features in the same order they are yielded by the dataset.
#   - Your model must have ≤ 5M trainable parameters (enforced during review).
# -----------------------------------------------------------------------------


# TODO: design and implement a better model
class AudioClassifier(nn.Module):
    def __init__(self, num_classes=15):
        super().__init__()

        self.flatten = nn.Flatten()
        # self.dropout = nn.Dropout(p=0.1)
        self.feature = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 128x431 → 64x215

            nn.Conv2d(8, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 64x215 → 32x107

            nn.Conv2d(16, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 32x107 → 16x53

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 16x53 → 8x26
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64*8*26, 256),  
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
        # We are not providing any example.
        # look back at lab 4 and add convolutional layers

    def forward(self, x):
        # same here, look back at lab 4.
        # x = self.flatten(x)
        # input ([batch, 1, 128, 431])
        x = self.feature(x)
        x = self.flatten(x) #need flatten to pass the classifier as it use linear
        x = self.classifier(x)

        return x
    