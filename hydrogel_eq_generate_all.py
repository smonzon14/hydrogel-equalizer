"""
Author: Sebastian Monzon
Date: 2024-06-10
Description: Generate songs processed with the smoothed EQ, evaluate their scores, and save the songs and scores in separate folders for each gain setting.
  The folders will be named according to the gain value used. Each folder will contain the processed audio files and a CSV file with the scores before and after EQ processing, as well as the improvement in score (delta).
"""

from hydrogel_eq import apply_eq_with_curve, read_wave_file, smoothed_eq, centers
from GTZAN.process_features import extract_bands_dev
import os
import numpy as np
import torch
from tqdm import tqdm

from GTZAN.standardized_features_to_score_dataset import StandardizedFeaturesToScoreDataset
from GTZAN.score_estimator import ScoreEstimatorModel
from scipy.io.wavfile import write

import pandas as pd

# Load the pre-trained score estimator model
score_estimator = ScoreEstimatorModel(input_size=32, hidden_sizes=[32,16])
checkpoint = torch.load("GTZAN/models/score_estimator_bands_dev_nyquist.pth", map_location='cpu')
score_estimator.load_state_dict(checkpoint)
score_estimator.eval()

gtzan_songs_dir = "GTZAN/genres_original"
output_base_dir = "hydrogel_eq_generated_songs"
if not os.path.exists(output_base_dir):
    os.makedirs(output_base_dir)

features_file_path = "GTZAN/bands_dev_global_features_nyquist.csv"
scores_file_path = "GTZAN/original_song_scores.csv"

original_dataset = StandardizedFeaturesToScoreDataset(
    features_csv=features_file_path,
    scores_csv=scores_file_path,
    outliers_txt="GTZAN/outliers.txt",
    ignore_columns=[0,1],  # Ignore filename
)
# original_dataset.analytical_normalization(feature_index=original_dataset.feature_columns.index("TotalRMS"))

merged_data = original_dataset.to_df()

# rename E_x_y columns to x - y in merged_data and correlation_matrix
# def rename_column(col_name):
#     if col_name.startswith("E"):
#         parts = col_name.split('_')
#         if len(parts) == 3:
#             return f"{parts[1]} Hz - {parts[2]} Hz"
#     return col_name
# merged_data.rename(columns=rename_column, inplace=True)

# sort merged_data by "score" column
merged_data.sort_values(by="score", ascending=False, inplace=True)
top_10 = merged_data.head(10)
top_10_filenames = top_10["filename"].tolist()
# Define gain and drive settings to evaluate
gain_settings = [3.0, 6.0, 9.0, 12.0, 15.0, 18.0]
drive = 3.1

scores_before = []
scores_after = {
    gain: [] for gain in gain_settings
}

full_song_filepaths = []
for song in top_10_filenames:
    full_song_filepaths.append(os.path.join(gtzan_songs_dir, song.split('.')[0], song))

pbar = tqdm(total=len(full_song_filepaths), desc=f"Processing original songs...", leave=True)
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
    with torch.no_grad():
        score_before = score_estimator(torch.tensor(features_before).unsqueeze(0))
        # score_before = torch.sigmoid(score_before).item()
        score_before = score_before.item()
        scores_before.append((filename, score_before))
    pbar.update(1)
pbar.close()


avg_delta_per_gain = {}

for gain in gain_settings:
    output_dir = os.path.join(output_base_dir, f"{int(gain)}dB")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    scores_after[gain] = []
    
    print(f"Processing songs with Gain: {gain} dB, Drive: {drive}")
    for idx, (filename, score_before) in enumerate(scores_before):
        file_path = full_song_filepaths[idx]
        try:
            audio, sr = read_wave_file(file_path)
            processed_audio, diff_db = apply_eq_with_curve(audio, sr, smoothed_eq, centers, G_max_db=gain, drive=drive)
            features_after = extract_bands_dev(y=processed_audio, sr=sr)
            features_after = np.array([features_after[field] for field in features_after.keys()], dtype=np.float32)
            features_after = original_dataset.normalize_features(np.array([features_after]))
            with torch.no_grad():
                score_after = score_estimator(torch.tensor(features_after).unsqueeze(0))
                # score_after = torch.sigmoid(score_after).item()
                score_after = score_after.item()
                delta = score_after - score_before
                sigmoid_delta = torch.sigmoid(torch.tensor(delta)).item()
            scores_after[gain].append((filename, score_before, score_after, sigmoid_delta))
            
            # Save processed audio
            # output_filepath = os.path.join(output_dir, filename)
            # write(output_filepath, sr, (processed_audio * 32767).astype(np.int16))
            # print(f"Saved processed {output_filepath}")
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
    
    # Calculate average delta for this gain setting
    deltas = [entry[3] for entry in scores_after[gain]]
    avg_delta = sum(deltas) / len(deltas) if deltas else 0
    avg_delta_per_gain[gain] = avg_delta
    print(f"Average Improvement for Gain {gain} dB: {avg_delta:.4f}")
    # Save scores to CSV
    scores_df = pd.DataFrame(scores_after[gain], columns=["filename", "Original Score (Predicted)", "Score After EQ (Predicted)", "Improvement"])
    scores_csv_path = os.path.join(output_dir, "scores.csv")
    scores_df.to_csv(scores_csv_path, index=False)
    print(f"Saved scores to {scores_csv_path}")
# Print summary of average deltas
print("Summary of Average Improvements per Gain Setting:")
for gain, avg_delta in avg_delta_per_gain.items():
    print(f"Gain {gain} dB: Average Improvement = {avg_delta:.4f}")

#plot average delta vs gain
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
import numpy as np
gains = list(avg_delta_per_gain.keys())
avg_deltas = list(avg_delta_per_gain.values())

# Add (0, 0) to the beginning of the lists
gains = gains
avg_deltas = avg_deltas
plt.figure(figsize=(4, 3))
gains_np = np.array(gains)
avg_deltas_np = np.array(avg_deltas)
xnew = np.linspace(gains_np.min(), gains_np.max(), 200)
spl = make_interp_spline(gains_np, avg_deltas_np, k=3)
y_smooth = spl(xnew)
plt.plot(xnew, y_smooth, marker='', linestyle='-', color='tab:blue')
plt.scatter(gains, avg_deltas, color='tab:blue', zorder=5)  # Show original points
# Plot text labels, and adjust limits to avoid clipping
# for x, y in zip(gains, avg_deltas):
#     label = f"({int(x)}, {y:.4f})"
#     if x == 0:
#         label = f"(0, 0)"
#     plt.text(x, y - 0.001, label, fontsize=9, ha='left', va='top')

# Expand x and y limits to prevent text clipping
plt.xlim(gains_np.min() - 1, gains_np.max() + 3)
plt.xticks(np.arange(gains_np.min(), gains_np.max() + 3, 3))
plt.ylim(min(avg_deltas_np) - 0.003, max(avg_deltas_np) + 0.003)
plt.xlabel("Equalizer Gain (dB)")
plt.ylabel("Logistic Comparative Score (σ(z₂-z₁))")
plt.tight_layout()
plt.savefig(os.path.join(output_base_dir, "average_improvement_vs_gain.png"), dpi=150)
print(f"Saved to {os.path.join(output_base_dir, 'average_improvement_vs_gain.png')}")