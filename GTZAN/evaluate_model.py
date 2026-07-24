import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import torch
import numpy as np
from torch.utils.data import Subset
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from standardized_features_to_score_dataset import StandardizedFeaturesToScoreDataset
from score_estimator import ScoreEstimatorModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

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
val_indices = list(range(split_idx, len(dataset)))
val_dataset = Subset(dataset, val_indices)

input_size = len(dataset.feature_columns)
model = ScoreEstimatorModel(input_size=input_size, output_size=1, hidden_sizes=[32, 16]).to(device)
model.load_state_dict(torch.load("GTZAN/models/score_estimator_bands_dev_nyquist.pth", map_location=device))
model.eval()

with torch.no_grad():
    val_features = torch.stack([val_dataset[i][0] for i in range(len(val_dataset))]).to(device)
    val_scores   = torch.stack([val_dataset[i][1] for i in range(len(val_dataset))]).to(device)
    val_outputs  = torch.sigmoid(model(val_features))

y_pred = val_outputs.squeeze().cpu().numpy()
y_true = val_scores.squeeze().cpu().numpy()

pearson_r, _   = pearsonr(y_pred, y_true)
spearman_rho, _ = spearmanr(y_pred, y_true)
r2             = r2_score(y_true, y_pred)
rmse           = np.sqrt(mean_squared_error(y_true, y_pred))
mae            = mean_absolute_error(y_true, y_pred)

print(f"\n{'='*40}")
print(f"  Validation set size : {len(val_dataset)} samples")
print(f"  Pearson r           : {pearson_r:.4f}")
print(f"  Spearman rho        : {spearman_rho:.4f}")
print(f"  R²                  : {r2:.4f}")
print(f"  RMSE                : {rmse:.4f}")
print(f"  MAE                 : {mae:.4f}")
print(f"{'='*40}\n")
