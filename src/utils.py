import random
from typing import List, Tuple
import torch
from torch.utils.data import Subset
from sklearn.model_selection import train_test_split
import numpy as np

def stratified_split(dataset, train_frac=0.8, val_frac=0.1, test_frac=0.1, seed=42):
    # X = indices of samples, Y = corresponding labels
    X = np.arange(len(dataset.samples))
    Y = np.array([dataset.label_to_idx[label] for _, label in dataset.samples])

    # First split: train vs rest
    X_train, X_rest, Y_train, Y_rest = train_test_split(
        X, Y, stratify=Y, train_size=train_frac, random_state=seed, shuffle=True
    )
    # Compute relative fraction for val/test from remaining
    rel = val_frac / (val_frac + test_frac)
    X_val, X_test, Y_val, Y_test = train_test_split(
        X_rest, Y_rest, stratify=Y_rest, train_size=rel, random_state=seed, shuffle=True
    )

    return X_train.tolist(), Y_train.tolist(), X_val.tolist(), Y_val.tolist(), X_test.tolist(), Y_test.tolist()

def simple_collate(batch):
    # batch: list of (waveform, label)
    waveforms = torch.stack([b[0] for b in batch], dim=0)  # [B, 1, N]
    labels = torch.stack([b[1] for b in batch], dim=0)
    return waveforms, labels