"""
Download the GTZAN audio dataset from Kaggle and place the genre folders under
GTZAN/genres_original/ (e.g. GTZAN/genres_original/blues/blues.00000.wav).

Run from the repository root:
    python GTZAN/download_dataset.py

The dataset is public, so no Kaggle account or API token is required.
"""
import os
import shutil
import sys

import kagglehub

KAGGLE_DATASET = "andradaolteanu/gtzan-dataset-music-genre-classification"
TARGET_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "genres_original")
EXPECTED_GENRES = ["blues", "classical", "country", "disco", "hiphop",
                   "jazz", "metal", "pop", "reggae", "rock"]


def count_wavs(directory):
    return sum(
        1
        for genre in EXPECTED_GENRES
        if os.path.isdir(os.path.join(directory, genre))
        for f in os.listdir(os.path.join(directory, genre))
        if f.endswith(".wav")
    )


def find_genres_original(root):
    """Locate the genres_original folder inside the downloaded dataset."""
    for dirpath, dirnames, _ in os.walk(root):
        if "genres_original" in dirnames:
            return os.path.join(dirpath, "genres_original")
    return None


def download_gtzan(force=False):
    if not force and count_wavs(TARGET_DIR) >= 999:
        print(f"GTZAN already present at {TARGET_DIR} ({count_wavs(TARGET_DIR)} wav files). Use --force to re-download.")
        return TARGET_DIR

    print(f"Downloading {KAGGLE_DATASET} (~1.3 GB)...")
    download_path = kagglehub.dataset_download(KAGGLE_DATASET, force_download=force)
    source_dir = find_genres_original(download_path)
    if source_dir is None:
        raise FileNotFoundError(f"Could not find a genres_original folder under {download_path}")

    print(f"Copying {source_dir} -> {TARGET_DIR}")
    shutil.copytree(source_dir, TARGET_DIR, dirs_exist_ok=True)

    n_wavs = count_wavs(TARGET_DIR)
    missing = [g for g in EXPECTED_GENRES if not os.path.isdir(os.path.join(TARGET_DIR, g))]
    if missing:
        raise RuntimeError(f"Download finished but genres are missing: {missing}")
    print(f"Done: {n_wavs} wav files across {len(EXPECTED_GENRES)} genres in {TARGET_DIR}")
    return TARGET_DIR


if __name__ == "__main__":
    download_gtzan(force="--force" in sys.argv)
