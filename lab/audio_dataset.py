import os
import csv
import torch
from torch.utils.data import Dataset
import torchaudio  
import random
from pathlib import Path



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
        self.root_dir = Path(root_dir)
        self.meta_path = self.root_dir / meta_filename
        self.audio_dir = self.root_dir / audio_subdir
        self.training_flag = training_flag

        # 2) Scan audio_subdir for candidate files.
        self.samples = []
        with open(self.meta_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t")
            # 3) Read metadata: filename + label string → keep only valid files.
            for row in reader:
                if len(row) >= 2:
                    filename = row[0]   # ex: "audio/b020_90_100.wav"
                    label = row[1]      # ex: "beach"
                    filepath = self.root_dir / filename
                    if filepath.exists():
                        # 5) Store samples as list of (filepath, label_string).
                        self.samples.append((filepath, label))
        
        # 4) Construct `self.class_names` (sorted unique labels) and then
        #    `self.label_to_idx` (class_name → integer index).
        unique_labels = sorted(set(label for _, label in self.samples))

        self.num_classes = len(unique_labels)
        self.class_names = unique_labels

        self.label_to_idx = {class_name: idx for idx, class_name in enumerate(self.class_names)}

        # 5) Audio processing parameters
        self.target_length = 220500  # 5 seconds at 44.1kHz
        self.sample_rate = 44100
        
        # 6) Mel-Spectrogram transform
        self.mel_spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=2048,        # FFT window size
            hop_length=512,    # Step size between frames
            n_mels=128,        # Number of mel frequency bins
            f_min=20,          # Minimum frequency
            f_max=8000,        # Maximum frequency (Nyquist = 22050)
        )

        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB()

    def __len__(self):
        return len(self.samples)

    def _load_and_process(self, path: str) -> torch.Tensor:
        waveform, sr = torchaudio.load(path)   # [channels, samples]
        # convert to mono if channels > 1
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        # resample if needed
        if sr != self.target_length:
            waveform = torchaudio.transforms.Resample(sr, self.target_length)(waveform)
        # pad or crop to target_length
        if waveform.shape[1] < self.target_length:
            pad = self.target_length - waveform.shape[1]
            waveform = torch.nn.functional.pad(waveform, (0, pad))
        elif waveform.shape[1] > self.target_length:
            start = 0
            if self.training_flag:
                start = random.randint(0, waveform.shape[1] - self.target_length)
            #example : target_length=20000, waveform.shape[1]=30000, if start=5000 by random then after crop get 5000 to 25000
            waveform = waveform[:, start:start + self.target_length]
        return waveform
    
    def _apply_augmentation(self, waveform):
        """
        Apply LIGHT data augmentation to waveform (only during training)
        
        Augmentations:
        - Time shift: Randomly shift audio in time (reduced)
        - Gaussian noise: Add small random noise (reduced)
        - Random gain: Adjust volume (reduced)
        """
        # REDUCED time shift (±0.2 seconds instead of ±0.5)
        # Too much shifting destroys temporal patterns
        max_shift = int(0.2 * self.sample_rate)
        shift = random.randint(-max_shift, max_shift)
        waveform = torch.roll(waveform, shift, dims=1)
        
        # REDUCED Gaussian noise (SNR ~40-50 dB, very subtle)
        noise_factor = random.uniform(0.001, 0.003)
        noise = torch.randn_like(waveform) * noise_factor
        waveform = waveform + noise
        
        # REDUCED random gain (±2 dB instead of ±3 dB)
        gain_db = random.uniform(-2, 2)
        gain = 10 ** (gain_db / 20)
        waveform = waveform * gain
        
        # Clip to prevent overflow
        waveform = torch.clamp(waveform, -1.0, 1.0)
        
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
        if self.training_flag:
            waveform = self._apply_augmentation(waveform)


        mel_spec = self.mel_spectrogram(waveform)  # [1, n_mels, time]
        mel_spec_db = self.amplitude_to_db(mel_spec)  # Convert to decibels

        # ensure tensor dtype float32
        waveform = waveform.float()
        label_idx = self.label_to_idx[label_str]
         # return waveform tensor and label tensor (label last)
        return mel_spec_db, label_idx