import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import torch
import numpy as np
from torch.utils.data import Subset
from scipy.stats import spearmanr, pearsonr
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from standardized_features_to_score_dataset import StandardizedFeaturesToScoreDataset

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
train_indices = list(range(0, split_idx))
val_indices   = list(range(split_idx, len(dataset)))
train_dataset = Subset(dataset, train_indices)
val_dataset   = Subset(dataset, val_indices)

X_train = torch.stack([train_dataset[i][0] for i in range(len(train_dataset))]).cpu().numpy()
y_train = torch.stack([train_dataset[i][1] for i in range(len(train_dataset))]).squeeze().cpu().numpy()
X_val   = torch.stack([val_dataset[i][0]   for i in range(len(val_dataset))]).cpu().numpy()
y_val   = torch.stack([val_dataset[i][1]   for i in range(len(val_dataset))]).squeeze().cpu().numpy()

param_grid = {
    "C":       [0.1, 1, 10, 100],
    "epsilon": [0.01, 0.05, 0.1, 0.2],
    "kernel":  ["rbf", "linear"],
}
grid = GridSearchCV(SVR(), param_grid, cv=5, scoring="r2", n_jobs=-1, verbose=1)
grid.fit(X_train, y_train)

print(f"\nBest params: {grid.best_params_}")
model = grid.best_estimator_
y_pred = model.predict(X_val)

pearson_r,   _ = pearsonr(y_pred, y_val)
spearman_rho,_ = spearmanr(y_pred, y_val)
r2             = r2_score(y_val, y_pred)
rmse           = np.sqrt(mean_squared_error(y_val, y_pred))
mae            = mean_absolute_error(y_val, y_pred)

print(f"\n{'='*40}")
print(f"  Validation set size : {len(val_dataset)} samples")
print(f"  Pearson r           : {pearson_r:.4f}")
print(f"  Spearman rho        : {spearman_rho:.4f}")
print(f"  R²                  : {r2:.4f}")
print(f"  RMSE                : {rmse:.4f}")
print(f"  MAE                 : {mae:.4f}")
print(f"{'='*40}\n")
