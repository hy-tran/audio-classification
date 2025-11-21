# audio-classification

## Dataset setup

Use the helper script in `scripts/download_data.py` to fetch the Audio Scene
Classification dataset and unpack it into `data/raw/audiosceneclassification2025`.

```bash
python scripts/download_data.py
```

The script is idempotent—it will reuse the existing archive or extracted folder
unless you pass `--force-download` and/or `--force-extract`. The downloaded
archive is saved under `data/raw/audiosceneclassification2025.zip`.

## Listen to a sample clip

Preview a random clip (or specify one) along with its label:

```bash
python scripts/listen_random_clip.py
# or specify a particular clip
python scripts/listen_random_clip.py --clip a001_50_60.wav
```

The script looks inside `data/raw/audiosceneclassification2025/audio` and uses
the metadata in `meta.txt` to show the label/clip ID.

## Training a model
we cannot feed  an audio file .wav directly to a model. We need to convert it into a tensor first (waveform).
```
waveform, sr = torchaudio.load("audio.wav")
waveform.shape  # torch.Size([num_channels, num_samples]) the number of channels and number of samples on each channel

print(waveform[:, :10])   # first 10 samples of each channel
tensor([[ 0.12, 0.50, 0.80, 0.20, -0.10, -0.50, -0.80, -0.30, 0.10, 0.40]])
```
For consistency, we will convert to mono channel (single channel) for any audio channels >1
Then we will pad or crop the waveform to a fixed length (e.g., 10 seconds) to ensure consistent input size for the model. (Maybe not needed if all audio files are already of the same time length. But for unexpected cases or for audio augmentation later, it's good to have this step.)
