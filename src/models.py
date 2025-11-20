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
    def __init__(self, num_classes: int):
        super().__init__()
        self.conv1d = nn.Conv1d(1, 50, kernel_size=5, stride=2, padding=2)
        self.pool   = nn.AdaptiveAvgPool1d(1)
        self.output   = nn.Linear(50, num_classes)


    def forward(self, feature1):
        x = feature1
        x = self.conv1d(x)    # Single conv layer
        x = self.pool(x)      # Reduce temporal dimension to 1
        x = x.squeeze(-1)     # [B, C, 1] -> [B, C]
        x = self.output(x)    # [B, num_classes]
        return x