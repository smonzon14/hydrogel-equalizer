"""
Generate control-EQ variants of the songs the hydrogel EQ is evaluated on (top 10 by score),
to test that the effect is specific to the discovered bands rather than to any EQ of the same strength.

Variants (all through the same apply_eq_with_curve path, same gain and drive):
  - baseline : 0 dB (processing only: STFT round trip, RMS matching, soft clipping)
  - optimal  : the hydrogel EQ
  - inverted : the hydrogel EQ with its sign flipped (boosts become cuts and vice versa)
  - random_k : the hydrogel EQ's band gains shuffled across bands (same gain values, wrong bands)

WAVs are written for the inverted and random variants only; baseline and optimal are scored for comparison.

Outputs:
  hydrogel_eq_generated_songs/controls_<gain>dB/<variant>/<song>.wav   (not committed)
  hydrogel_eq_generated_songs/control_eq_curves.csv / .png
  hydrogel_eq_generated_songs/control_eq_scores.csv   (MLP-predicted scores)
"""
from hydrogel_eq import apply_eq_with_curve, read_wave_file, smoothed_eq, centers
from GTZAN.process_features import extract_bands_dev, low_bands, med_bands, high_bands
from GTZAN.standardized_features_to_score_dataset import StandardizedFeaturesToScoreDataset
from GTZAN.score_estimator import ScoreEstimatorModel
import os
import numpy as np
import pandas as pd
import torch
import soundfile as sf
import matplotlib.pyplot as plt
from tqdm import tqdm

GAIN = 15.0  # dB
DRIVE = 3.1  # soft clipping drive
N_RANDOM = 3
SEED = 0
MAX_ABS_SIMILARITY = 0.2  # reject shuffles that happen to resemble the optimal (or inverted) curve
SCORE_ONLY_VARIANTS = {"baseline", "optimal"}  # scored for comparison, but no WAVs written (already exist)

gtzan_songs_dir = "GTZAN/genres_original"
output_base_dir = "hydrogel_eq_generated_songs"
wav_dir = os.path.join(output_base_dir, f"controls_{int(GAIN)}dB")

# --- EQ curves ---
def random_curves(eq_curve, n, seed, max_abs_similarity):
    """Seeded permutations of eq_curve across bands, kept only if |Pearson r| to eq_curve is small."""
    rng = np.random.default_rng(seed)
    curves = []
    while len(curves) < n:
        candidate = rng.permutation(eq_curve)
        if abs(np.corrcoef(candidate, eq_curve)[0, 1]) < max_abs_similarity:
            curves.append(candidate)
    return curves

# name -> (curve in [-1, 1], gain in dB)
variants = {
    "baseline": (smoothed_eq, 0.0),
    "optimal": (smoothed_eq, GAIN),
    "inverted": (-smoothed_eq, GAIN),
}
for k, curve in enumerate(random_curves(smoothed_eq, N_RANDOM, SEED, MAX_ABS_SIMILARITY), start=1):
    variants[f"random_{k}"] = (curve, GAIN)

band_labels = [f"{lo} Hz - {hi} Hz" for lo, hi in np.concatenate([low_bands, med_bands, high_bands])]
curves_df = pd.DataFrame({"Frequency Band": band_labels})
for name, (curve, gain) in variants.items():
    curves_df[f"{name} (dB)"] = curve * gain
curves_csv_path = os.path.join(output_base_dir, "control_eq_curves.csv")
curves_df.to_csv(curves_csv_path, index=False)
print(f"Saved curves to {curves_csv_path}")
for name, (curve, gain) in variants.items():
    if gain > 0:
        print(f"  {name}: similarity to optimal (Pearson r) = {np.corrcoef(curve, smoothed_eq)[0, 1]:+.3f}")

fig, axs = plt.subplots(len(variants) - 1, 1, figsize=(6, 1.6 * (len(variants) - 1)), sharex=True, sharey=True)
x = np.arange(len(smoothed_eq))
for ax, (name, (curve, gain)) in zip(axs, [(n, v) for n, v in variants.items() if v[1] > 0]):
    eq_db = curve * gain
    ax.plot(x, eq_db, color="black", linewidth=1.0)
    ax.fill_between(x, eq_db, 0, where=(eq_db >= 0), interpolate=True, color="#2878B8", alpha=0.45)
    ax.fill_between(x, eq_db, 0, where=(eq_db < 0), interpolate=True, color="#B43426", alpha=0.45)
    ax.axhline(0, color="gray", linestyle="--", linewidth=1)
    ax.set_ylabel(name)
    ax.set_xticks([])
axs[-1].set_xlabel("Frequency (Hz)")
fig.supylabel("Equalizer Gain (dB)")
fig.tight_layout()
fig.savefig(os.path.join(output_base_dir, "control_eq_curves.png"), dpi=300)

# --- Score model + top-10 songs, same as hydrogel_eq_generate_all.py ---
score_estimator = ScoreEstimatorModel(input_size=32, hidden_sizes=[32, 16])
score_estimator.load_state_dict(torch.load("GTZAN/models/score_estimator_bands_dev_nyquist.pth", map_location="cpu"))
score_estimator.eval()

original_dataset = StandardizedFeaturesToScoreDataset(
    features_csv="GTZAN/bands_dev_global_features_nyquist.csv",
    scores_csv="GTZAN/original_song_scores.csv",
    outliers_txt="GTZAN/outliers.txt",
    ignore_columns=[0, 1],  # Ignore filename, label
)
merged_data = original_dataset.to_df()
merged_data.sort_values(by="score", ascending=False, inplace=True)
top_10_filenames = merged_data.head(10)["filename"].tolist()

def predict_score(audio, sr):
    features = extract_bands_dev(y=audio, sr=sr)
    features = np.array([features[field] for field in features.keys()], dtype=np.float32)
    features = original_dataset.normalize_features(np.array([features]))
    with torch.no_grad():
        return score_estimator(torch.tensor(features).unsqueeze(0)).item()

# --- Generate ---
rows = []
pbar = tqdm(total=len(top_10_filenames) * len(variants), desc="Generating control EQs", leave=True)
for filename in top_10_filenames:
    audio, sr = read_wave_file(os.path.join(gtzan_songs_dir, filename.split('.')[0], filename))
    assert sr == 22050, "Expected 22.05 kHz sample rate."
    row = {"filename": filename, "original": predict_score(audio, sr)}
    for name, (curve, gain) in variants.items():
        processed_audio, _ = apply_eq_with_curve(audio, sr, curve, centers, G_max_db=gain, drive=DRIVE)
        if name not in SCORE_ONLY_VARIANTS:
            os.makedirs(os.path.join(wav_dir, name), exist_ok=True)
            sf.write(os.path.join(wav_dir, name, filename), processed_audio, sr, subtype="PCM_16")
        row[name] = predict_score(processed_audio, sr)
        pbar.update(1)
    rows.append(row)
pbar.close()

# Predicted score (logit) per variant, and logistic comparative score vs the 0 dB baseline: σ(z_variant - z_baseline)
scores_df = pd.DataFrame(rows)
for name in variants:
    if name != "baseline":
        scores_df[f"{name} vs baseline"] = torch.sigmoid(torch.tensor((scores_df[name] - scores_df["baseline"]).values)).numpy()
scores_csv_path = os.path.join(output_base_dir, "control_eq_scores.csv")
scores_df.to_csv(scores_csv_path, index=False)
print(f"Saved WAVs to {wav_dir}/<variant>/ and scores to {scores_csv_path}")
print("Mean logistic comparative score vs 0 dB baseline (0.5 = no change):")
for name in variants:
    if name != "baseline":
        print(f"  {name}: {scores_df[f'{name} vs baseline'].mean():.4f}")
