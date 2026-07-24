import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, to_rgba
from matplotlib.ticker import MaxNLocator
from standardized_features_to_score_dataset import StandardizedFeaturesToScoreDataset

# Custom "coolwarm" variant: keep coolwarm's red half, but transition the blue
# half toward #105BA4 instead of the default purple-blue.
# Reversed so negative values map to red and positive values map to blue.
_coolwarm = plt.cm.coolwarm
_center = np.array(_coolwarm(0.5))          # light gray midpoint
_blue = np.array(to_rgba("#2878B8"))
_blue_half = [tuple(c) for c in np.linspace(_blue, _center, 128)]
_red_half = [_coolwarm(x) for x in np.linspace(0.5, 1.0, 128)]
CUSTOM_COOLWARM = LinearSegmentedColormap.from_list(
    "custom_coolwarm", _red_half[::-1] + _blue_half[::-1]
)

METHOD = 'spearman'  # 'pearson' or 'spearman'

# Load the datasets
features_file_path = "GTZAN/bands_dev_global_features_nyquist.csv"
feature_type = "Power Density"

# features_file_path = "GTZAN/bands_energy_features_nyquist.csv"
# feature_type = "RMS Energy"

scores_file_path = "GTZAN/original_song_scores.csv"


dataset = StandardizedFeaturesToScoreDataset(
    features_csv=features_file_path,
    scores_csv=scores_file_path,
    outliers_txt="GTZAN/outliers.txt",
    ignore_columns=[0,1],  # Ignore filename
)
dataset.analytical_normalization(feature_index=dataset.feature_columns.index("TotalRMS"))
merged_data = dataset.to_df()
merged_data = merged_data.drop(columns=["label", "filename"])
print("Dataset shape after normalization:", merged_data.shape)
print("Dataset columns:", merged_data.columns.tolist())
# Drop the song name column
# Filter features to include only columns starting with "EDevFrac"
# merged_data = merged_data.loc[:, merged_data.columns.str.startswith("EDevFrac") | (merged_data.columns == "score")]
#square the column "TotalRMS"
# merged_data['TotalPower'] = merged_data['TotalRMS'] ** 2
# Compute the Spearman correlation matrix
correlation_matrix = merged_data.corr(method=METHOD)

# Sort the other columns by their correlation to "score"
sorted_columns = correlation_matrix['score'].drop('score').abs().sort_values(ascending=False).index

# Select the top 10 columns with the highest correlation to "Water loss (score)"
top_columns = ['score'] + sorted_columns[:10].tolist()

# Filter the correlation matrix to only include the top 10 columns
correlation_matrix = correlation_matrix.loc[top_columns, top_columns]

letter = 'ρ' if METHOD == 'spearman' else 'r'

# rename E_x_y columns to x - y in merged_data and correlation_matrix
def rename_column(col_name):
    if col_name == "score":
        return "Water loss (score)"
    if col_name == "TotalRMS":
        return "Total RMS"
    if col_name.startswith("E"):
        parts = col_name.split('_')
        if len(parts) == 3:
            return f"{parts[1]}Hz - {parts[2]}Hz"
    return col_name
merged_data.rename(columns=rename_column, inplace=True)
correlation_matrix.rename(index=rename_column, columns=rename_column, inplace=True)


# Plot the correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", cbar_kws={'label': f'Correlation ({letter})'}, cbar=True, center=0)
# plt.title(f"{METHOD.capitalize()} Correlation Matrix")
# tilt x-axis labels for better readability
plt.xticks(rotation=45, ha='right')
# ensure padding so labels are not cut off
plt.tight_layout()
plt.show()
plt.savefig("GTZAN/frequency_band_correlation_matrix.png", dpi=300)

# plot each feature as a bar chart with its correlation to score. Leave original ordering
plt.figure(figsize=(10, 6))
correlations = merged_data.corr(method=METHOD)['Water loss (score)'].drop('Water loss (score)')

# Normalize the correlation values for color mapping
norm = plt.Normalize(correlations.min(), correlations.max())
colors = CUSTOM_COOLWARM(norm(correlations))

# Plot the bar chart with color-coded bars
correlations.plot(kind='bar', color=colors)
plt.ylabel(f"Correlation of Band {feature_type} to Water loss ({letter})")
# plt.title(f"Correlation of Band Power Density (Deviation from Mean) to Water Extraction (%)")
# plt.text(
#     0.5, 0.9, 
#     r"Note: Band Power Density = $\frac{\mathrm{RMS}^2}{f_{\mathrm{hi}} - f_{\mathrm{low}}}$ (where $f_{\mathrm{low}}$ and $f_{\mathrm{hi}}$ are in Hz)", 
#     ha='center', va='center', transform=plt.gca().transAxes, 
#     fontsize=12, color='gray'
# )
plt.xticks(rotation=45, ha='right')
plt.xlabel(r"Frequency Band (${f_{low}}$ - ${f_{hi}}$)")
# plt.xlabel(r"Band Power Deviation Fraction (EDevFrac_${\mathit{f_{low}}}$_${\mathit{f_{hi}}}$)")

plt.tight_layout()
plt.show()

plt.savefig("GTZAN/frequency_band_correlations.png", dpi=300)

# Compact version for embedding in a larger figure: smaller, tighter bars,
# minimal labeling.
plt.figure(figsize=(3.9, 2.5))
correlations.plot(kind='bar', color=colors, width=0.95)
plt.ylabel(f"Correlation to Score ({letter})")
plt.xlabel("Frequency (Hz)")
# Drop the per-band x tick labels (precise labeling not needed here)
plt.xticks([])
# Fewer y ticks
plt.gca().yaxis.set_major_locator(MaxNLocator(nbins=4))
plt.tight_layout()
plt.show()
plt.savefig("GTZAN/frequency_band_correlations_compact.png", dpi=300)

# Save the correlation values to a CSV file
# correlations.to_csv("GTZAN/frequency_band_spearman_correlations.csv", header=['spearman_correlation'])
