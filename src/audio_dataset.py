import os
import csv
import random
import torch
from torch.utils.data import Dataset
import torchaudio  
from typing import List, Tuple

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
    def __init__(self, root_dir, meta_filename, audio_subdir, training_flag: bool = True,
                 target_sr: int = 22050,
                 duration: float = 10.0):
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

        self.root_dir = root_dir
        self.meta_path = os.path.join(root_dir, meta_filename)
        self.audio_dir = os.path.join(root_dir, audio_subdir)
        self.training_flag = training_flag
        self.target_sr = target_sr
        self.num_samples = int(target_sr * duration)

        # read metadata CSV (expects columns: filename,label)
        samples: List[Tuple[str,str]] = []
        labels = []
        with open(self.meta_path, newline='', encoding='utf-8') as f:
            rdr = csv.reader(f)
            for row in rdr:
                if not row: 
                    continue
                fname, label = row[0].strip(), row[1].strip()
                audio_path = os.path.join(self.audio_dir, fname)
                if os.path.isfile(audio_path):
                    samples.append((audio_path, label))
                    labels.append(label)
        if not samples:
            raise RuntimeError(f"No valid audio files found in {self.audio_dir} using {self.meta_path}")
        
        # build class_names and mapping
        class_names = sorted(list(set(labels)))
        self.class_names = class_names
        self.label_to_idx = {c: i for i, c in enumerate(class_names)}
        self.num_classes = len(class_names)
        self.samples = samples

        # resampler used when sr != target_sr
        self.resampler = torchaudio.transforms.Resample(orig_freq=44100, new_freq=self.target_sr)

    def __len__(self):
        return len(self.samples)

    def _load_and_process(self, path: str) -> torch.Tensor:
        waveform, sr = torchaudio.load(path)   # [channels, samples]
        # convert to mono if channels > 1
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        # resample if needed
        if sr != self.target_sr:
            waveform = torchaudio.transforms.Resample(sr, self.target_sr)(waveform)
        # pad or crop to num_samples
        if waveform.shape[1] < self.num_samples:
            pad = self.num_samples - waveform.shape[1]
            waveform = torch.nn.functional.pad(waveform, (0, pad))
        elif waveform.shape[1] > self.num_samples:
            start = 0
            if self.training_flag:
                start = random.randint(0, waveform.shape[1] - self.num_samples)
            #example : num_samples=20000, waveform.shape[1]=30000, if start=5000 by random then after crop get 5000 to 25000
            waveform = waveform[:, start:start + self.num_samples]
        return waveform

    def __getitem__(self, idx):
        # Steps to implement:
        #   1) filepath, label_str = self.samples[idx]
        #   2) waveform, sr = torchaudio.load(filepath)
        #   3) If self.training_flag is True, apply augmentations here
        #   4) Ensure waveform shape is consistent (crop/pad if necessary)
        #   5) label_idx = self.label_to_idx[label_str]
        #   6) Return (feature(s), label_idx) with the label as the final item.

        path, label_str = self.samples[idx]
        waveform = self._load_and_process(path)
        # if self.training_flag:
        #     # apply augmentations here

        # ensure tensor dtype float32
        waveform = waveform.float()
        label_idx = self.label_to_idx[label_str]
         # return waveform tensor and label tensor (label last)
        return waveform, torch.tensor(label_idx, dtype=torch.long)