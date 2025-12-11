import torch
from torch.utils.data import DataLoader, Subset
from ..src.audio_dataset import LoadAudio  # your dataset class
from sklearn.model_selection import train_test_split
import numpy as np

# -----------------------------
# Load dataset
# -----------------------------
dataset = LoadAudio(
    root_dir="data",
    meta_filename="meta.csv",
    audio_subdir="audio",
    training_flag=True,
    target_sr=220500,
    duration=10.0
)

print(f"Total samples: {len(dataset)}")
print(f"Classes: {dataset.class_names}")