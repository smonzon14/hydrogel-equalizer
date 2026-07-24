import sys
import os
sys.path.insert(0, "GTZAN")

from hydrogel_eq import apply_eq_with_curve, read_wave_file, smoothed_eq, centers
from GTZAN.process_features import extract_bands_dev
from GTZAN.standardized_features_to_score_dataset import StandardizedFeaturesToScoreDataset
import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
from scipy.stats import spearmanr
from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error, r2_score
from torch.utils.data import Subset

# --- Train LASSO on same split as svm_estimator.py ---
torch.manual_seed(42)
dataset = StandardizedFeaturesToScoreDataset(
    features_csv="GTZAN/bands_dev_global_features_nyquist.csv",
    scores_csv="GTZAN/original_song_scores.csv",
    outliers_txt="GTZAN/outliers.txt",
    ignore_columns=[0, 1],
    normalize=True,
    feature_indeces=None,
    shuffle=True,
    shuffle_seed=4,
    pca_components=None,
)
split_idx = int(0.8 * len(dataset))
train_dataset = Subset(dataset, list(range(0, split_idx)))
val_dataset = Subset(dataset, list(range(split_idx, len(dataset))))
X_train = torch.stack([train_dataset[i][0] for i in range(len(train_dataset))]).cpu().numpy()
y_train = torch.stack([train_dataset[i][1] for i in range(len(train_dataset))]).squeeze().cpu().numpy()
X_val = torch.stack([val_dataset[i][0] for i in range(len(val_dataset))]).cpu().numpy()
y_val = torch.stack([val_dataset[i][1] for i in range(len(val_dataset))]).squeeze().cpu().numpy()

print("Training LASSO...")
param_grid = {
    "alpha": [0.0001, 0.001, 0.01, 0.1, 1.0],
}
grid = GridSearchCV(Lasso(max_iter=10000), param_grid, cv=5, scoring="r2", n_jobs=-1, verbose=0)
grid.fit(X_train, y_train)
lasso = grid.best_estimator_
print(f"Best LASSO params: {grid.best_params_}")

# --- Validation metrics ---
y_pred = lasso.predict(X_val)
print(f"Validation Spearman rho: {spearmanr(y_val, y_pred).correlation:.4f}")
print(f"Validation R^2: {r2_score(y_val, y_pred):.4f}")
print(f"Validation MAE: {mean_absolute_error(y_val, y_pred):.4f}")

def lasso_score(features):
    pred = float(np.clip(lasso.predict(features)[0], 1e-6, 1 - 1e-6))
    return pred

def logit(p):
    return np.log(p / (1 - p))

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# --- Dataset for normalization (no shuffle, matches generate_all) ---
original_dataset = StandardizedFeaturesToScoreDataset(
    features_csv="GTZAN/bands_dev_global_features_nyquist.csv",
    scores_csv="GTZAN/original_song_scores.csv",
    outliers_txt="GTZAN/outliers.txt",
    ignore_columns=[0, 1],
    normalize=True,
    shuffle=False,
    pca_components=None,
)

# --- Select top 10 songs by score, same as generate_all ---
merged_data = original_dataset.to_df()
merged_data.sort_values(by="score", ascending=False, inplace=True)
top_10_filenames = merged_data.head(10)["filename"].tolist()

gtzan_songs_dir = "GTZAN/genres_original"
full_song_filepaths = [
    os.path.join(gtzan_songs_dir, song.split('.')[0], song) for song in top_10_filenames
]

# --- Compute scores_before for top 10 ---
scores_before = []
pbar = tqdm(total=len(full_song_filepaths), desc="Processing original songs...", leave=True)
for filename, file_path in zip(top_10_filenames, full_song_filepaths):
    if not filename.endswith(".wav"):
        continue
    try:
        audio, sr = read_wave_file(file_path)
        assert sr == 22050, "Expected 22.05 kHz sample rate."
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        continue
    features_before = extract_bands_dev(y=audio, sr=sr)
    features_before = np.array([features_before[field] for field in features_before.keys()], dtype=np.float32)
    features_before = original_dataset.normalize_features(np.array([features_before]))
    score_before = lasso_score(features_before)
    scores_before.append((filename, score_before))
    pbar.update(1)
pbar.close()

# --- Evaluate each gain setting ---
gain_settings = [0, 1.0, 2.0, 3.0, 6.0, 9.0, 12.0, 15.0, 18.0]
drive = 3.1
scores_after = {gain: [] for gain in gain_settings}
avg_delta_per_gain = {}

for gain in gain_settings:
    print(f"Processing songs with Gain: {gain} dB, Drive: {drive}")
    for idx, (filename, score_before) in enumerate(scores_before):
        file_path = full_song_filepaths[idx]
        try:
            audio, sr = read_wave_file(file_path)
            processed_audio, diff_db = apply_eq_with_curve(audio, sr, smoothed_eq, centers, G_max_db=gain, drive=drive)
            features_after = extract_bands_dev(y=processed_audio, sr=sr)
            features_after = np.array([features_after[field] for field in features_after.keys()], dtype=np.float32)
            features_after = original_dataset.normalize_features(np.array([features_after]))
            score_after = lasso_score(features_after)
            # Logistic comparative score: sigmoid(logit(after) - logit(before))
            sigmoid_delta = sigmoid(logit(score_after) - logit(score_before))
            scores_after[gain].append((filename, score_before, score_after, sigmoid_delta))
        except Exception as e:
            print(f"Error processing {file_path}: {e}")

    deltas = [entry[3] for entry in scores_after[gain]]
    avg_delta = sum(deltas) / len(deltas) if deltas else 0
    avg_delta_per_gain[gain] = avg_delta
    print(f"Average Improvement for Gain {gain} dB: {avg_delta:.4f}")

# --- Plot ---
gains = list(avg_delta_per_gain.keys())
avg_deltas = list(avg_delta_per_gain.values())
plt.figure(figsize=(4, 3))
gains_np = np.array(gains)
avg_deltas_np = np.array(avg_deltas)
xnew = np.linspace(gains_np.min(), gains_np.max(), 200)
spl = make_interp_spline(gains_np, avg_deltas_np, k=3)
y_smooth = spl(xnew)
plt.plot(xnew, y_smooth, marker='', linestyle='-', color='tab:blue')
plt.scatter(gains, avg_deltas, color='tab:blue', zorder=5)
plt.xlim(gains_np.min() - 1, gains_np.max() + 3)
plt.xticks(np.arange(gains_np.min(), gains_np.max() + 3, 3))
plt.ylim(min(avg_deltas_np) - 0.003, max(avg_deltas_np) + 0.003)
plt.xlabel("Equalizer Gain (dB)")
plt.ylabel("Logistic Comparative Score (σ(z₂-z₁))")
plt.tight_layout()
plt.savefig("hydrogel_eq_generated_songs/average_improvement_vs_gain_lasso.png", dpi=150)
print("Saved to hydrogel_eq_generated_songs/average_improvement_vs_gain_lasso.png")
