import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.stats import spearmanr
import os


class StandardizedFeaturesToScoreDataset(Dataset):
    def __init__(
        self,
        features_csv=os.path.join(os.path.dirname(__file__), "features_30_sec.csv"),
        scores_csv=os.path.join(os.path.dirname(__file__), "original_song_scores.csv"),
        outliers_txt=os.path.join(os.path.dirname(__file__), "outliers.txt"),
        ignore_columns=[
            0,
            1,
            -1,
        ],  # Ignore filename, length, and label columns (for GTZAN)
        normalize=True,
        feature_indeces=None,
        shuffle=True,
        shuffle_seed=42,
        n_mel_features=None,
        pca_components=None,
        analytical_normalization_index=None,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Read scores CSV, skip header row
        scores_df = pd.read_csv(scores_csv, skiprows=1, header=None)
        # Remove outliers if outliers_txt is provided
        if outliers_txt and os.path.exists(outliers_txt):
            with open(outliers_txt, "r") as f:
                outliers = set(line.strip() for line in f.readlines())
            scores_df = scores_df[~scores_df.iloc[:, 0].isin(outliers)].reset_index(
                drop=True
            )

            # Standardize scores to [0, 1]
            min_score = scores_df.iloc[:, 1].min()
            max_score = scores_df.iloc[:, 1].max()
            scores_df.iloc[:, 1] = (scores_df.iloc[:, 1] - min_score) / (
                max_score - min_score + 1e-8
            )

        # Tensors cannot be used as dicts directly. Instead, keep the dict as is, but you can convert the scores to tensors if needed:
        self.score_dict = {
            k: torch.tensor([v], dtype=torch.float32).to(self.device)
            for k, v in zip(scores_df.iloc[:, 0], scores_df.iloc[:, 1])
        }

        # Store the first column (filename) for score lookup
        self.filenames = list(self.score_dict.keys())
        # Read features CSV, skip header row
        features_df = pd.read_csv(
            features_csv, skiprows=0, header=0
        )  # Read with headers

        # Ensure negative indices are handled correctly
        num_columns = features_df.shape[1]
        ignore_columns = [
            col if col >= 0 else num_columns + col for col in ignore_columns
        ]
        self.feature_columns = [
            col for i, col in enumerate(features_df.columns) if i not in ignore_columns
        ]
        # Filter rows where the filename (first column) is not in self.filenames
        features_df = features_df[
            features_df.iloc[:, 0].isin(self.filenames)
        ].reset_index(drop=True)
        if shuffle:
            features_df = features_df.sample(
                frac=1, random_state=shuffle_seed
            ).reset_index(drop=True)
        # Select the desired columns (3rd to second to last)
        print(f"Using {len(self.feature_columns)} feature columns.")
        self.features = features_df.loc[:, self.feature_columns].to_numpy(
            dtype=np.float32
        )

        # Apply analytical normalization if specified
        if analytical_normalization_index is not None:
            self.analytical_normalization(analytical_normalization_index)

        # Min-max normalization to [0, 1] for each feature column
        self.mins = self.features.min(axis=0)
        self.maxs = self.features.max(axis=0)
        if normalize:
            self.features = self.normalize_features(self.features)
        # Update filenames to match filtered features_df
        self.filenames = features_df.iloc[:, 0].tolist()

        if n_mel_features:
            # Generate and append advanced features (mel spectrum averages)
            mel_features_list = []
            for i in range(len(self.features)):
                mel_feat = self.generate_mel_spectrum_average_features(
                    i, n_mels=n_mel_features
                )
                mel_features_list.append(mel_feat)
            mel_features = np.stack(mel_features_list, axis=0)
            mel_mins = mel_features.min(axis=0)
            mel_maxs = mel_features.max(axis=0)
            mel_features = (mel_features - mel_mins) / (mel_maxs - mel_mins + 1e-8)
            # Concatenate with existing features
            self.features = np.concatenate([self.features, mel_features], axis=1)
            self.feature_columns += [
                f"mel_band_{i+1}_mean" for i in range(n_mel_features)
            ]

        if pca_components:
            # Apply PCA to reduce dimensionality
            self.pca = PCA(n_components=pca_components)
            features_np = self.features
            features_pca = self.pca.fit_transform(features_np)
            self.features = np.concatenate([self.features, features_pca], axis=1)
            self.feature_columns += [
                f"pca_component_{i+1}" for i in range(pca_components)
            ]

        # If feature_indeces is provided, filter self.features to only those columns
        if feature_indeces is not None:
            self.features = self.features[:, feature_indeces]
            self.feature_columns = [self.feature_columns[i] for i in feature_indeces]

        self.features = torch.tensor(self.features, dtype=torch.float32).to(self.device)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        feature = self.features[idx]
        filename = self.filenames[idx]
        score = self.score_dict.get(
            filename, torch.tensor([0.0], dtype=torch.float32).to(self.device)
        )
        return feature, score, filename

    def analytical_normalization(self, feature_index):
        """Apply analytical normalization to remove confounding factor."""
        values = self.features[:, feature_index].cpu().numpy()
        mean_value = np.mean(values)
        # mean_score = torch.mean(torch.cat([self.score_dict[f] for f in self.filenames])).item()
        # Fit a 2nd degree polynomial: score = a * RMS^2 + b * RMS + c
        coeffs = np.polyfit(
            values,
            torch.cat([self.score_dict[f] for f in self.filenames]).cpu().numpy(),
            2,
        )
        a, b, c = coeffs
        print(
            f"Removing confounding factor for feature {self.feature_columns[feature_index]} using 2nd degree polynomial fit."
        )
        print(f"Analytical normalization coefficients: a={a}, b={b}, c={c}")
        analytical_mean_score = a * mean_value**2 + b * mean_value + c
        print(
            f"Analytical mean score at mean value {mean_value:.3f}: {analytical_mean_score:.3f}"
        )
        # Apply normalization: normalized_score = score - (a*RMS^2 + b*RMS + c) + mean_score
        normalized_scores = []
        for i, f in enumerate(self.filenames):
            val = values[i].item()
            original_score = self.score_dict[f].item()
            normalized_score = (
                original_score - (a * val**2 + b * val + c) + analytical_mean_score
            )
            normalized_scores.append(normalized_score)

        # standardize normalized scores to [0, 1]
        normalized_scores = np.array(normalized_scores)
        min_score = normalized_scores.min()
        max_score = normalized_scores.max()
        normalized_scores = (normalized_scores - min_score) / (max_score - min_score)
        # convert back to tensor and update score_dict
        for i, f in enumerate(self.filenames):
            self.score_dict[f] = torch.tensor(
                [normalized_scores[i]], dtype=torch.float32
            ).to(self.device)
        # Remove the confounding feature from features
        self.features = torch.cat(
            [self.features[:, :feature_index], self.features[:, feature_index + 1 :]],
            dim=1,
        )
        self.feature_columns.pop(feature_index)

    def get_wav_path(self, idx):
        """Get the original WAV file path for the given index."""
        filename = self.filenames[idx]
        return os.path.join(
            os.path.dirname(__file__),
            "genres_original",
            filename.split(".")[0],
            filename,
        )

    def generate_mel_spectrum_average_features(self, idx, n_mels=512):
        """Generate mel spectrum average features for the given index."""
        import librosa
        import os

        filename = self.filenames[idx]
        wav_path = os.path.join(
            os.path.dirname(__file__),
            "genres_original",
            filename.split(".")[0],
            filename,
        )
        y, sr = librosa.load(wav_path, sr=None)
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
        S_dB = librosa.power_to_db(S, ref=np.max)
        # Compute mean and variance across time for each mel band
        mel_means = S_dB.mean(axis=1)
        # mel_vars = S_dB.var(axis=1)
        # mel_features = np.concatenate([mel_means, mel_vars]).astype(np.float32)
        return mel_means.astype(np.float32)

    def get_mean_score(self):
        """Calculate the mean score across all entries."""
        all_scores = (
            torch.cat([self.score_dict[f] for f in self.filenames])
            .cpu()
            .numpy()
            .flatten()
        )
        return np.mean(all_scores)

    def normalize_features(self, features):
        """Normalize features to [0, 1] using the stored mins and maxs."""
        if features.shape[1] > len(self.mins):
            normalized = (features[:, : len(self.mins)] - self.mins) / (
                self.maxs - self.mins + 1e-8
            )
            return np.concatenate([normalized, features[:, len(self.mins) :]], axis=1)
        else:
            return (features - self.mins) / (self.maxs - self.mins + 1e-8)

    def get_pca_components(self, features, concatenate=True):
        """Apply PCA transformation to the given features."""
        if hasattr(self, "pca"):
            pca_features = self.pca.transform(features)
            if pca_features.shape[1] == 0:
                return features
            if concatenate:
                return np.concatenate([features, pca_features], axis=1)
        else:
            raise ValueError(
                "PCA has not been fitted. Initialize the dataset with pca_components parameter."
            )

    def get_top_k_items_by_score(self, k=5):
        """Get the top k items by score."""
        all_scores = (
            torch.cat([self.score_dict[f] for f in self.filenames])
            .cpu()
            .numpy()
            .flatten()
        )
        top_k_indices = np.argsort(all_scores)[-k:][::-1]
        top_k_items = [(self.filenames[i], all_scores[i]) for i in top_k_indices]
        return top_k_items

    def to_df(self):
        """Convert the dataset to a pandas DataFrame."""
        data = {
            "filename": self.filenames,
            "score": [self.score_dict[f].item() for f in self.filenames],
            "label": [f.split(".")[0] for f in self.filenames],
        }
        for i, col in enumerate(self.feature_columns):
            data[col] = self.features[:, i].cpu().numpy()

        return pd.DataFrame(data)


if __name__ == "__main__":
    dataset = StandardizedFeaturesToScoreDataset(
        features_csv="GTZAN/bands_dev_global_features_nyquist.csv",
        scores_csv="GTZAN/original_song_scores.csv",
        outliers_txt="GTZAN/outliers.txt",
        ignore_columns=[0, 1],
    )
    print(f"Dataset length: {len(dataset)}")
    print(f"Dataset columns: {dataset.feature_columns}")

    # do a test of analytical normalization on the top feature
    dataset.analytical_normalization(feature_index=-1)
    mean_score_after = dataset.get_mean_score()
    print("Top 5 items by score after normalization:")
    top_5 = dataset.get_top_k_items_by_score(k=5)
    for fname, sc in top_5:
        print(f"{fname}: {sc:.4f}")

    features, score, filename = dataset[2]

    print(f"Feature shape: {features.shape}, Score: {score}, Filename: {filename}")
    # print(f"Feature columns: {dataset.feature_columns}")
    # print(f"Feature {dataset.feature_columns[0]} value: {feature[0].item()}")
    # Stack all features and scores
    all_features = dataset.features.cpu().numpy()  # type: ignore
    all_scores = (
        torch.cat([dataset.score_dict[f] for f in dataset.filenames])
        .cpu()
        .numpy()
        .flatten()
    )

    # Compute correlation of each feature with the score
    correlations = []
    for i, col in enumerate(dataset.feature_columns):
        corr, _ = spearmanr(all_features[:, i], all_scores)
        correlations.append((col, corr))

    # Sort by absolute correlation value, descending
    correlations.sort(key=lambda x: abs(x[1]), reverse=True)

    N = 30

    n_features_to_show = min(N, len(correlations))
    print(f"Top {n_features_to_show} features by absolute correlation with score:")
    columns = []
    for col, corr in correlations[:N]:
        print(f"{col}: {corr:.4f}")
        columns.append(dataset.feature_columns.index(col))

    # scaler = StandardScaler()
    # features_scaled = scaler.fit_transform(all_features)

    # Optionally, perform PCA and print explained variance
    pca = PCA(n_components=n_features_to_show)
    pca.fit_transform(all_features)
    print("\nPCA explained variance ratios:", pca.explained_variance_ratio_)
    print(
        f"Total explained variance by first {n_features_to_show} components:",
        np.sum(pca.explained_variance_ratio_),
    )

    print("\nUsing feature indices:", columns)
