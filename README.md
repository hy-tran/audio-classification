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
python -i scripts/listen_random_clip.py
# or specify a particular clip
python -i scripts/listen_random_clip.py --clip a001_50_60.wav
```

The script looks inside `data/raw/audiosceneclassification2025/audio` and uses
the metadata in `meta.txt` to show the label/clip ID.
