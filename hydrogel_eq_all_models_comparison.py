"""
Overlay of the EQ-gain sweep (logistic comparative score vs gain) for every model:
MLP, Ridge, Lasso, linear SVR, RBF SVR. All models are evaluated on identical
audio-derived features and the same gain grid.
"""
import sys
import os
sys.path.insert(0, "GTZAN")

from hydrogel_eq import apply_eq_with_curve, read_wave_file, smoothed_eq, centers
from GTZAN.process_features import extract_bands_dev
from GTZAN.standardized_features_to_score_dataset import StandardizedFeaturesToScoreDataset
from GTZAN.score_estimator import ScoreEstimatorModel
import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
from sklearn.svm import SVR
from sklearn.linear_model import Ridge, Lasso
from sklearn.model_selection import GridSearchCV
from torch.utils.data import Subset

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def logit(p):
    return np.log(p / (1 - p))

# --- Training split (same as the per-model scripts) ---
torch.manual_seed(42)
dataset = StandardizedFeaturesToScoreDataset(
    features_csv="GTZAN/bands_dev_global_features_nyquist.csv",
    scores_csv="GTZAN/original_song_scores.csv",
    outliers_txt="GTZAN/outliers.txt",
    ignore_columns=[0, 1],
    normalize=True, shuffle=True, shuffle_seed=4, pca_components=None,
)
split_idx = int(0.8 * len(dataset))
train_dataset = Subset(dataset, list(range(0, split_idx)))
X_train = torch.stack([train_dataset[i][0] for i in range(len(train_dataset))]).cpu().numpy()
y_train = torch.stack([train_dataset[i][1] for i in range(len(train_dataset))]).squeeze().cpu().numpy()

# --- Build all models. Each entry maps name -> predict_proba(features_row)->(0,1) ---
def fit_svr(kernel):
    grid = GridSearchCV(SVR(), {"C": [0.1, 1, 10, 100], "epsilon": [0.01, 0.05, 0.1, 0.2], "kernel": [kernel]},
                        cv=5, scoring="r2", n_jobs=-1)
    grid.fit(X_train, y_train)
    return grid.best_estimator_

print("Training sklearn models...")
ridge = GridSearchCV(Ridge(), {"alpha": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]}, cv=5, scoring="r2", n_jobs=-1).fit(X_train, y_train).best_estimator_
lasso = GridSearchCV(Lasso(max_iter=10000), {"alpha": [0.0001, 0.001, 0.01, 0.1, 1.0]}, cv=5, scoring="r2", n_jobs=-1).fit(X_train, y_train).best_estimator_
svr_linear = fit_svr("linear")
svr_rbf = fit_svr("rbf")

mlp = ScoreEstimatorModel(input_size=32, hidden_sizes=[32, 16])
mlp.load_state_dict(torch.load("GTZAN/models/score_estimator_bands_dev_nyquist.pth", map_location="cpu"))
mlp.eval()

def sk_proba(model):
    return lambda feats: float(np.clip(model.predict(feats)[0], 1e-6, 1 - 1e-6))

def mlp_proba(feats):
    with torch.no_grad():
        z = mlp(torch.tensor(feats, dtype=torch.float32)).item()
    return float(np.clip(sigmoid(z), 1e-6, 1 - 1e-6))

# name -> (proba_fn, color, linestyle)
MODELS = {
    "MLP":          (mlp_proba,        "black",      "-"),
    "Ridge":        (sk_proba(ridge),  "tab:green",  "-"),
    "Lasso":        (sk_proba(lasso),  "tab:purple", "-"),
    "SVR (linear)": (sk_proba(svr_linear), "tab:blue", "-"),
    "SVR (RBF)†": (sk_proba(svr_rbf), "tab:red",  "--"),  # excluded; dashed (dagger resolved in caption)
}

# --- Normalization dataset + top-10 songs ---
original_dataset = StandardizedFeaturesToScoreDataset(
    features_csv="GTZAN/bands_dev_global_features_nyquist.csv",
    scores_csv="GTZAN/original_song_scores.csv",
    outliers_txt="GTZAN/outliers.txt",
    ignore_columns=[0, 1], normalize=True, shuffle=False, pca_components=None,
)
merged = original_dataset.to_df().sort_values(by="score", ascending=False)
top_10 = merged.head(10)["filename"].tolist()
paths = [os.path.join("GTZAN/genres_original", s.split('.')[0], s) for s in top_10]

gain_settings = [0, 1.0, 2.0, 3.0, 6.0, 9.0, 12.0, 15.0, 18.0]
drive = 3.1

def feats_of(audio, sr):
    f = extract_bands_dev(y=audio, sr=sr)
    f = np.array([f[k] for k in f.keys()], dtype=np.float32)
    return original_dataset.normalize_features(np.array([f]))

# --- Extract features once (model-independent) ---
before, after = {}, {g: {} for g in gain_settings}
pbar = tqdm(total=len(paths) * (len(gain_settings) + 1), desc="Extracting features", leave=True)
for idx, p in enumerate(paths):
    audio, sr = read_wave_file(p)
    before[idx] = feats_of(audio, sr)
    pbar.update(1)
    for g in gain_settings:
        proc, _ = apply_eq_with_curve(audio, sr, smoothed_eq, centers, G_max_db=g, drive=drive)
        after[g][idx] = feats_of(proc, sr)
        pbar.update(1)
pbar.close()

# --- Compute comparative-score curve per model ---
curves = {}
for name, (proba, _, _) in MODELS.items():
    p_before = {idx: proba(before[idx]) for idx in before}
    curve = []
    for g in gain_settings:
        deltas = [sigmoid(logit(proba(after[g][idx])) - logit(p_before[idx])) for idx in before]
        curve.append(np.mean(deltas))
    curves[name] = np.array(curve)
    print(f"{name}: " + ", ".join(f"{g}dB={v:.4f}" for g, v in zip(gain_settings, curves[name])))

# --- Overlay plot ---
gains_np = np.array(gain_settings)
xnew = np.linspace(gains_np.min(), gains_np.max(), 300)
plt.figure(figsize=(7, 5))
for name, (_, color, ls) in MODELS.items():
    y = curves[name]
    y_smooth = make_interp_spline(gains_np, y, k=3)(xnew)
    plt.plot(xnew, y_smooth, ls, color=color, label=name)
    plt.scatter(gains_np, y, color=color, s=25, zorder=5)
plt.axhline(0.5, color="gray", linewidth=0.9, linestyle=":", label="no change (0.5)")
plt.axvline(15, color="gray", linewidth=0.8, alpha=0.5)
plt.xlabel("Equalizer Gain (dB)")
plt.ylabel("Logistic Comparative Score (σ(z₂-z₁))")
plt.title("Predicted improvement vs EQ gain, by model")
plt.legend()
plt.tight_layout()
plt.savefig("hydrogel_eq_generated_songs/all_models_improvement_vs_gain.png", dpi=150)
print("Saved to hydrogel_eq_generated_songs/all_models_improvement_vs_gain.png")

# --- Write the underlying data to CSV (one row per gain, one column per model) ---
import csv
csv_path = "hydrogel_eq_generated_songs/all_models_improvement_vs_gain.csv"
model_names = list(MODELS.keys())
with open(csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["gain_db"] + model_names)
    for i, g in enumerate(gain_settings):
        writer.writerow([g] + [f"{curves[name][i]:.6f}" for name in model_names])
print(f"Saved to {csv_path}")
