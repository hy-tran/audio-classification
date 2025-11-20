import os
import csv
import torch
from torch.utils.data import Dataset
import torchaudio  

# -----------------------------------------------------------------------------
# DATASET IMPLEMENTATION NOTES
#
# - The dataset may return multiple features (e.g., waveform, spectrogram,
#   loudness estimate, embeddings, etc.).
#
# - All returned features must have consistent shapes within a batch.
#   Different feature types may have different shapes from each other, but the
#   *same* feature type must have the *same shape* across all samples in the batch.
#   (Example: if feature1 is a raw waveform, each feature 1 should be 
#   padded/cropped to N samples.
#
# - If applying random transforms (noise, time-shift, gain, etc.), only do so
#   when `self.training_flag` is True to ensure evaluation is deterministic.
#
# REQUIREMENTS FOR EVALUATION:
# We will test you dataset. Ensure the following:
#   1) The label is the last returned item. 
#      e.g. return feature1, feature2, label
#   2) All returned items are PyTorch tensors.
#   3) `self.class_names` contains the sorted unique class names.
#   4) Any augmentation or preprocessing happens inside the dataset, not in
#      external training/evaluation loops.
# -----------------------------------------------------------------------------


class LoadAudio(Dataset):
    def __init__(self, root_dir, meta_filename, audio_subdir, training_flag: bool = True):
        """
        Args:
            root_dir (str): Dataset root directory.
            meta_filename (str): Metadata filename inside root_dir.
            audio_subdir (str): Audio subdirectory relative to root_dir.
            training_flag (bool): When True, random transforms may be applied
                                  inside __getitem__ for data augmentation.
        """

        # 1) Store the directories/paths.
        # 2) Scan audio_subdir for candidate files.
        # 3) Read metadata: filename + label string → keep only valid files.
        # 4) Construct `self.class_names` (sorted unique labels) and then
        #    `self.label_to_idx` (class_name → integer index).
        # 5) Store samples as list of (filepath, label_string).

        self.training_flag = training_flag
        self.samples = [None] * 300               # list of (filepath, label_string), change this
        self.num_classes = 15                 # placeholder
        self.class_names = ["placeholder"] * self.num_classes
        self.label_to_idx = {}                # fill as {class_name: index}

        # Temporary placeholder waveform size; remove when real loading implemented.
        self.waveform_shape = (1, 220500)     # ~5 seconds at ~44.1kHz


    def __len__(self):
        return len(self.samples)


    def __getitem__(self, idx):
        # Steps to implement:
        #   1) filepath, label_str = self.samples[idx]
        #   2) waveform, sr = torchaudio.load(filepath)
        #   3) If self.training_flag is True, apply augmentations here
        #   4) Ensure waveform shape is consistent (crop/pad if necessary)
        #   5) label_idx = self.label_to_idx[label_str]
        #   6) Return (feature(s), label_idx) with the label as the final item.

        # Placeholder output for now
        dummy_waveform = torch.randn(self.waveform_shape)

        # if self.training_flag:
        #     # apply augmentations here

        label = torch.randint(0, self.num_classes, (1,)).item()
        return dummy_waveform, label