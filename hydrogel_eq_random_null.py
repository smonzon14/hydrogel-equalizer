"""
Null distribution for the hydrogel EQ: predicted improvement for many random EQs (the hydrogel EQ's
band gains shuffled across bands, unscreened) vs the optimal and inverted EQ, for every model.
Same songs (top 10 by score), gain and drive as hydrogel_eq_generate_controls.py; all comparisons
are against the 0 dB processed baseline.

Outputs (in hydrogel_eq_generated_songs/):
  random_null_distribution.csv  one row per EQ curve: similarity to optimal + score per model
  random_null_distribution.png  null histograms per model with optimal / inverted marked
"""
import sys
import os
sys.path.insert(0, "GTZAN")

from hydrogel_eq import apply_eq_with_curve, read_wave_file, smoothed_eq, centers
from GTZAN.process_features import extract_bands_dev, low_bands, med_bands, high_bands
from GTZAN.standardized_features_to_score_dataset import StandardizedFeaturesToScoreDataset
from GTZAN.score_estimator import ScoreEstimatorModel
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.linear_model import Ridge, Lasso
from sklearn.model_selection import GridSearchCV
from torch.utils.data import Subset

GAIN = 15.0  # dB
DRIVE = 3.1  # soft clipping drive
N_RANDOM = 100
SEED = 1
output_base_dir = "hydrogel_eq_generated_songs"

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def logit(p):
    return np.log(p / (1 - p))

# --- Models: same training split and grids as hydrogel_eq_all_models_comparison.py ---
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

MODELS = {
    "MLP": mlp_proba,
    "Ridge": sk_proba(ridge),
    "Lasso": sk_proba(lasso),
    "SVR (linear)": sk_proba(svr_linear),
    "SVR (RBF)†": sk_proba(svr_rbf),  # excluded model, shown for completeness
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

def feats_of(audio, sr):
    f = extract_bands_dev(y=audio, sr=sr)
    f = np.array([f[k] for k in f.keys()], dtype=np.float32)
    return original_dataset.normalize_features(np.array([f]))

# --- EQ curves: optimal, inverted, the 3 generated controls, and N_RANDOM unscreened shuffles ---
band_widths = np.array([hi - lo for lo, hi in np.concatenate([low_bands, med_bands, high_bands])], dtype=np.float64)

def weighted_corr(a, b, w):
    a_c, b_c = a - np.average(a, weights=w), b - np.average(b, weights=w)
    return np.sum(w * a_c * b_c) / np.sqrt(np.sum(w * a_c**2) * np.sum(w * b_c**2))

curves = {"optimal": smoothed_eq, "inverted": -smoothed_eq}
controls_csv = os.path.join(output_base_dir, "control_eq_curves.csv")
if os.path.exists(controls_csv):
    controls_df = pd.read_csv(controls_csv)
    for col in controls_df.columns:
        if col.startswith("random_"):
            curves[f"control_{col.split(' ')[0]}"] = controls_df[col].values / GAIN
rng = np.random.default_rng(SEED)
for k in range(N_RANDOM):
    curves[f"null_{k:03d}"] = rng.permutation(smoothed_eq)

# --- Extract features once per (song, curve); models are applied afterwards ---
baseline, after = {}, {name: {} for name in curves}
pbar = tqdm(total=len(paths) * (len(curves) + 1), desc="Extracting features", leave=True)
for idx, p in enumerate(paths):
    audio, sr = read_wave_file(p)
    proc, _ = apply_eq_with_curve(audio, sr, smoothed_eq, centers, G_max_db=0.0, drive=DRIVE)
    baseline[idx] = feats_of(proc, sr)
    pbar.update(1)
    for name, curve in curves.items():
        proc, _ = apply_eq_with_curve(audio, sr, curve, centers, G_max_db=GAIN, drive=DRIVE)
        after[name][idx] = feats_of(proc, sr)
        pbar.update(1)
pbar.close()

# --- Mean logistic comparative score vs baseline, per curve and model ---
rows = []
p_base = {m: {idx: proba(baseline[idx]) for idx in baseline} for m, proba in MODELS.items()}
for name, curve in curves.items():
    row = {
        "curve": name,
        "similarity (Pearson r, per band)": np.corrcoef(curve, smoothed_eq)[0, 1],
        "similarity (Pearson r, bandwidth-weighted)": weighted_corr(curve, smoothed_eq, band_widths),
    }
    for m, proba in MODELS.items():
        row[m] = np.mean([sigmoid(logit(proba(after[name][idx])) - logit(p_base[m][idx])) for idx in baseline])
    rows.append(row)
results = pd.DataFrame(rows)
csv_path = os.path.join(output_base_dir, "random_null_distribution.csv")
results.to_csv(csv_path, index=False)
print(f"Saved to {csv_path}")

# --- Summary + plot ---
null = results[results["curve"].str.startswith("null_")]
named = results.set_index("curve")
fig, axs = plt.subplots(1, len(MODELS), figsize=(3.2 * len(MODELS), 3.2), sharey=True)
print(f"\nMean logistic comparative score vs 0 dB baseline at {GAIN:g} dB (0.5 = no change), {len(null)} random EQs:")
for ax, m in zip(axs, MODELS):
    opt, inv = named.loc["optimal", m], named.loc["inverted", m]
    n_ge = int((null[m] >= opt).sum())
    p_value = (n_ge + 1) / (len(null) + 1)
    print(f"  {m:13s} optimal={opt:.4f} inverted={inv:.4f} | random: mean={null[m].mean():.4f} sd={null[m].std():.4f} "
          f"min={null[m].min():.4f} max={null[m].max():.4f} | random >= optimal: {n_ge}/{len(null)} (p={p_value:.3f}) "
          f"| random > 0.5: {int((null[m] > 0.5).sum())}/{len(null)}")
    ax.hist(null[m], bins=20, color="gray", alpha=0.6, label=f"random (n={len(null)})")
    ax.axvline(opt, color="#2878B8", linewidth=2, label="optimal")
    ax.axvline(inv, color="#B43426", linewidth=2, label="inverted")
    ax.axvline(0.5, color="gray", linewidth=0.9, linestyle=":")
    for c in [c for c in named.index if c.startswith("control_")]:
        ax.axvline(named.loc[c, m], color="black", linewidth=0.8, linestyle="--",
                   label="generated controls" if c.endswith("_1") else None)
    ax.set_title(f"{m}\n(p={p_value:.3f})", fontsize=10)
    ax.set_xlabel("Logistic Comparative Score")
axs[0].set_ylabel("Number of random EQs")
axs[0].legend(fontsize=7)
fig.tight_layout()
fig.savefig(os.path.join(output_base_dir, "random_null_distribution.png"), dpi=200)
print(f"Saved to {os.path.join(output_base_dir, 'random_null_distribution.png')}")
print("\nSpearman correlation between a random EQ's similarity to optimal and its score:")
for m in MODELS:
    print(f"  {m:13s} per-band r: {null['similarity (Pearson r, per band)'].corr(null[m], method='spearman'):+.3f}"
          f"   bandwidth-weighted r: {null['similarity (Pearson r, bandwidth-weighted)'].corr(null[m], method='spearman'):+.3f}")

# --- Generated control EQs (optimal, inverted, random_k): curve next to its predicted score per model ---
if os.path.exists(controls_csv):
    BOOST_BLUE, CUT_RED = "#2878B8", "#B43426"
    control_names = [c.split(" ")[0] for c in controls_df.columns if c.endswith("(dB)") and not c.startswith("baseline")]
    model_names = list(MODELS.keys())
    fig, axs = plt.subplots(len(control_names), 2, figsize=(11, 2.1 * len(control_names)), gridspec_kw={"width_ratios": [1.5, 1]})
    x = np.arange(len(controls_df))
    for name, (ax, bx) in zip(control_names, axs):
        row = named.loc[name if name in named.index else f"control_{name}"]
        eq_db = controls_df[f"{name} (dB)"].values
        ax.plot(x, eq_db, color="black", linewidth=1)
        ax.fill_between(x, eq_db, 0, where=(eq_db >= 0), interpolate=True, color=BOOST_BLUE, alpha=0.45)
        ax.fill_between(x, eq_db, 0, where=(eq_db < 0), interpolate=True, color=CUT_RED, alpha=0.45)
        ax.axhline(0, color="gray", linestyle="--", linewidth=1)
        ax.set_ylim(-GAIN - 2, GAIN + 2)
        ax.set_xticks([])
        ax.set_ylabel("Gain (dB)")
        ax.set_title(f"{name}  (similarity to optimal r = {row['similarity (Pearson r, per band)']:+.2f})", fontsize=10, loc="left")
        scores = row[model_names].values.astype(float)
        bx.bar(range(len(scores)), scores - 0.5, bottom=0.5, color=[BOOST_BLUE if s >= 0.5 else CUT_RED for s in scores], alpha=0.75, width=0.65)
        bx.axhline(0.5, color="gray", linestyle="--", linewidth=1)
        for i, s in enumerate(scores):
            bx.text(i, s + (0.012 if s >= 0.5 else -0.012), f"{s:.3f}", ha="center", va="bottom" if s >= 0.5 else "top", fontsize=8)
        bx.set_ylim(0.2, 0.62)
        bx.set_xticks(range(len(scores)))
        bx.set_xticklabels(model_names if name == control_names[-1] else [], fontsize=8)
        bx.set_ylabel("Predicted score")
    axs[0, 1].set_title("Predicted score vs 0 dB baseline (0.5 = no change)", fontsize=10, loc="left")
    axs[-1, 0].set_xlabel("Frequency band (0 Hz to 11 kHz, 31 bands)")
    fig.tight_layout()
    fig.savefig(os.path.join(output_base_dir, "control_eqs_with_scores.png"), dpi=180)
    print(f"Saved to {os.path.join(output_base_dir, 'control_eqs_with_scores.png')}")
