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
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error, r2_score
from torch.utils.data import Subset

KERNELS = ["linear", "rbf"]
KERNEL_COLORS = {"linear": "tab:blue", "rbf": "tab:red"}

def logit(p):
    return np.log(p / (1 - p))

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# --- Training split (same as svm_estimator.py) ---
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

# --- Train one SVR per kernel, record in-distribution (non-EQ'd) metrics ---
models = {}
for kernel in KERNELS:
    print(f"\nTraining SVM (kernel={kernel})...")
    param_grid = {
        "C":       [0.1, 1, 10, 100],
        "epsilon": [0.01, 0.05, 0.1, 0.2],
        "kernel":  [kernel],
    }
    grid = GridSearchCV(SVR(), param_grid, cv=5, scoring="r2", n_jobs=-1, verbose=0)
    grid.fit(X_train, y_train)
    models[kernel] = grid.best_estimator_
    print(f"Best params ({kernel}): {grid.best_params_}")

    y_pred = grid.best_estimator_.predict(X_val)
    print(f"[{kernel}] In-distribution Spearman rho: {spearmanr(y_val, y_pred).correlation:.4f}")
    print(f"[{kernel}] In-distribution R^2: {r2_score(y_val, y_pred):.4f}")
    print(f"[{kernel}] In-distribution MAE: {mean_absolute_error(y_val, y_pred):.4f}")

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

gain_settings = [0, 1.0, 2.0, 3.0, 6.0, 9.0, 12.0, 15.0, 18.0]
drive = 3.1

# --- Extract features ONCE (kernel-independent), cache for reuse ---
# features_before_cache[idx] = normalized feature row for the original song
# features_after_cache[(gain, idx)] = normalized feature row for the EQ'd song
features_before_cache = {}
features_after_cache = {}

pbar = tqdm(total=len(full_song_filepaths), desc="Extracting original features...", leave=True)
for idx, file_path in enumerate(full_song_filepaths):
    if not full_song_filepaths[idx].endswith(".wav"):
        continue
    try:
        audio, sr = read_wave_file(file_path)
        assert sr == 22050, "Expected 22.05 kHz sample rate."
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        continue
    feats = extract_bands_dev(y=audio, sr=sr)
    feats = np.array([feats[field] for field in feats.keys()], dtype=np.float32)
    features_before_cache[idx] = original_dataset.normalize_features(np.array([feats]))
    pbar.update(1)
pbar.close()

for gain in gain_settings:
    print(f"Extracting EQ'd features at Gain: {gain} dB, Drive: {drive}")
    for idx, file_path in enumerate(full_song_filepaths):
        try:
            audio, sr = read_wave_file(file_path)
            processed_audio, diff_db = apply_eq_with_curve(audio, sr, smoothed_eq, centers, G_max_db=gain, drive=drive)
            feats = extract_bands_dev(y=processed_audio, sr=sr)
            feats = np.array([feats[field] for field in feats.keys()], dtype=np.float32)
            features_after_cache[(gain, idx)] = original_dataset.normalize_features(np.array([feats]))
        except Exception as e:
            print(f"Error processing {file_path} at gain {gain}: {e}")

# --- For each kernel, predict over cached features and build the gain sweep ---
def predict(model, features):
    return float(np.clip(model.predict(features)[0], 1e-6, 1 - 1e-6))

avg_delta_per_gain_by_kernel = {}
for kernel in KERNELS:
    model = models[kernel]
    scores_before = {idx: predict(model, features_before_cache[idx]) for idx in features_before_cache}

    avg_delta_per_gain = {}
    for gain in gain_settings:
        deltas = []
        for idx in features_before_cache:
            if (gain, idx) not in features_after_cache:
                continue
            score_before = scores_before[idx]
            score_after = predict(model, features_after_cache[(gain, idx)])
            deltas.append(sigmoid(logit(score_after) - logit(score_before)))
        avg_delta_per_gain[gain] = sum(deltas) / len(deltas) if deltas else 0
        print(f"[{kernel}] Gain {gain} dB: avg improvement = {avg_delta_per_gain[gain]:.4f}")
    avg_delta_per_gain_by_kernel[kernel] = avg_delta_per_gain

# --- Per-kernel plots ---
def plot_curve(avg_delta_per_gain, color, out_path, title=None):
    gains = list(avg_delta_per_gain.keys())
    avg_deltas = list(avg_delta_per_gain.values())
    plt.figure(figsize=(4, 3))
    gains_np = np.array(gains)
    avg_deltas_np = np.array(avg_deltas)
    xnew = np.linspace(gains_np.min(), gains_np.max(), 200)
    spl = make_interp_spline(gains_np, avg_deltas_np, k=3)
    y_smooth = spl(xnew)
    plt.plot(xnew, y_smooth, linestyle='-', color=color)
    plt.scatter(gains, avg_deltas, color=color, zorder=5)
    plt.xlim(gains_np.min() - 1, gains_np.max() + 3)
    plt.xticks(np.arange(gains_np.min(), gains_np.max() + 3, 3))
    plt.ylim(min(avg_deltas_np) - 0.003, max(avg_deltas_np) + 0.003)
    plt.xlabel("Equalizer Gain (dB)")
    plt.ylabel("Logistic Comparative Score (σ(z₂-z₁))")
    if title:
        plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"Saved to {out_path}")

for kernel in KERNELS:
    plot_curve(
        avg_delta_per_gain_by_kernel[kernel],
        KERNEL_COLORS[kernel],
        f"hydrogel_eq_generated_songs/average_improvement_vs_gain_svm_{kernel}.png",
        title=f"SVM ({kernel}) — EQ'd (out-of-distribution)",
    )

# --- Overlay comparison plot ---
plt.figure(figsize=(5, 4))
for kernel in KERNELS:
    avg_delta_per_gain = avg_delta_per_gain_by_kernel[kernel]
    gains_np = np.array(list(avg_delta_per_gain.keys()))
    avg_deltas_np = np.array(list(avg_delta_per_gain.values()))
    xnew = np.linspace(gains_np.min(), gains_np.max(), 200)
    y_smooth = make_interp_spline(gains_np, avg_deltas_np, k=3)(xnew)
    plt.plot(xnew, y_smooth, linestyle='-', color=KERNEL_COLORS[kernel], label=f"{kernel}")
    plt.scatter(gains_np, avg_deltas_np, color=KERNEL_COLORS[kernel], zorder=5)
plt.axhline(0.5, color='black', linewidth=0.8, linestyle='--', label='no change (0.5)')
plt.xlabel("Equalizer Gain (dB)")
plt.ylabel("Logistic Comparative Score (σ(z₂-z₁))")
plt.title("SVM kernel comparison on EQ'd audio")
plt.legend()
plt.tight_layout()
plt.savefig("hydrogel_eq_generated_songs/average_improvement_vs_gain_svm_kernel_comparison.png", dpi=150)
print("Saved to hydrogel_eq_generated_songs/average_improvement_vs_gain_svm_kernel_comparison.png")
