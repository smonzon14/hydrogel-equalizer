from hydrogel_eq import *
from GTZAN.process_features import extract_bands_dev
from GTZAN.score_estimator import ScoreEstimatorModel
from GTZAN.standardized_features_to_score_dataset import StandardizedFeaturesToScoreDataset
import os
import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

score_estimator = ScoreEstimatorModel(input_size=32, hidden_sizes=[32,16])
checkpoint = torch.load("GTZAN/models/score_estimator_bands_dev_nyquist.pth", map_location='cpu')
score_estimator.load_state_dict(checkpoint)
score_estimator.eval()

dataset = StandardizedFeaturesToScoreDataset(
    features_csv="GTZAN/bands_dev_global_features_nyquist.csv",
    scores_csv="GTZAN/original_song_scores.csv",
    outliers_txt="GTZAN/outliers.txt",
    ignore_columns=[0,1],  # Ignore filename, label
    pca_components=None,
    normalize=True,
    shuffle=False,
)

gtzan_songs_dir = "GTZAN/genres_original/"

# Precompute features_before for each song and store in a list
song_filepaths = []
features_before_list = []
scores_before_list = []

print("Loading GTZAN songs and extracting features...")
for genre in os.listdir(gtzan_songs_dir):
    genre_dir = os.path.join(gtzan_songs_dir, genre)
    if not os.path.isdir(genre_dir):
        continue
    i = 0
    for filename in os.listdir(genre_dir):
        # if i >= 1:
        #     break
        i += 1
        if not filename.endswith(".wav"):
            continue
        file_path = os.path.join(genre_dir, filename)
        try:
            audio, sr = read_wave_file(file_path)
            assert sr == 22050, "Expected 22.05 kHz sample rate."
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            continue
        features_before = extract_bands_dev(y=audio, sr=sr)
        features_before = np.array([features_before[field] for field in features_before.keys()], dtype=np.float32)
        features_before = dataset.normalize_features(np.array([features_before]))
        song_filepaths.append((file_path, audio, sr))
        features_before_list.append(features_before)
        with torch.no_grad():
            score_before = score_estimator(torch.tensor(features_before).unsqueeze(0))
            score_before = torch.sigmoid(score_before).item()
            scores_before_list.append(score_before)
        if i == 1:
            pbar = tqdm(total=50, desc=f"Processing {genre}", leave=True)
        pbar.update(1)
pbar.close()

def evaluate_eq(G_max_db, drive, song_filepaths, features_before_list, scores_before_list):
    scores_after = []
    for idx, (file_path, audio, sr) in enumerate(song_filepaths):
        try:
            processed_audio, diff_db = apply_eq_with_curve(audio, sr, smoothed_eq, centers, G_max_db=G_max_db, drive=drive)
            features_after = extract_bands_dev(y=processed_audio, sr=sr)
            features_after = np.array([features_after[field] for field in features_after.keys()], dtype=np.float32)
            features_after = dataset.normalize_features(np.array([features_after]))
            with torch.no_grad():
                score_after = score_estimator(torch.tensor(features_after).unsqueeze(0))
                score_after = torch.sigmoid(score_after).item()
            scores_after.append(score_after)
            print(f"Processing {file_path}...")
            print(f"Score before EQ: {scores_before_list[idx]:.4f}, Score after EQ: {score_after:.4f}, Improvement: {score_after - scores_before_list[idx]:.4f}")
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            scores_after.append(scores_before_list[idx])
    avg_score_before = sum(scores_before_list) / len(scores_before_list)
    avg_score_after = sum(scores_after) / len(scores_after)
    print(f"Average Score before EQ: {avg_score_before:.4f}")
    print(f"Average Score after EQ: {avg_score_after:.4f}")
    print(f"Average Improvement: {avg_score_after - avg_score_before:.4f}")
    return scores_after

# Define the parameter grid
G_max_db_values = [20.0, 16.0, 12.0, 8.0]
drive_values = [0.5, 1.0, 1.5]
delta_matrix = np.zeros((len(G_max_db_values), len(drive_values)))

fig, axs = plt.subplots(len(G_max_db_values), len(drive_values), figsize=(15, 12))
axs = axs.flatten()

for idx, (G_max_db, drive) in enumerate([(g, d) for g in G_max_db_values for d in drive_values]):
    scores_after = evaluate_eq(G_max_db, drive, song_filepaths, features_before_list, scores_before_list)
    ax = axs[idx]
    ax.hist(scores_before_list, bins=30, alpha=0.5, label='Original')
    ax.hist(scores_after, bins=30, alpha=0.5, label='EQ Applied (gain={}, drive={})'.format(G_max_db, drive))
    ax.set_xlabel('Estimated Score')
    ax.set_ylabel('Number of Songs')
    ax.set_title(f'G_max_db={G_max_db}, drive={drive}')
    ax.legend(loc='upper right')
    # Store delta in matrix
    i = G_max_db_values.index(G_max_db)
    j = drive_values.index(drive)
    delta_matrix[i, j] = np.mean(scores_after) - np.mean(scores_before_list)

# Plot the delta matrix as a colored square matrix
fig2, ax2 = plt.subplots(figsize=(6, 5))
im = ax2.imshow(delta_matrix, cmap='RdYlGn', vmin=np.min(delta_matrix), vmax=np.max(delta_matrix))
for i in range(len(G_max_db_values)):
    for j in range(len(drive_values)):
        ax2.text(j, i, f"{delta_matrix[i, j]:.3f}", ha="center", va="center", color="black", fontsize=12)
ax2.set_xticks(np.arange(len(drive_values)))
ax2.set_yticks(np.arange(len(G_max_db_values)))
ax2.set_xticklabels(drive_values)
ax2.set_yticklabels(G_max_db_values)
ax2.set_xlabel("Drive")
ax2.set_ylabel("G_max_db")
ax2.set_title("Average Score Improvement (Delta)")
fig2.colorbar(im, ax=ax2)

plt.tight_layout()
plt.show()
import pdb; pdb.set_trace()
