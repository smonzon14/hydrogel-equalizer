import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from GTZAN.standardized_features_to_score_dataset import StandardizedFeaturesToScoreDataset
import wave
from scipy.ndimage import gaussian_filter1d
import numpy as np
from scipy.signal import stft, istft

METHOD = 'spearman'  # 'pearson' or 'spearman'

# Load the datasets
features_file_path = "GTZAN/bands_dev_global_features_nyquist.csv"
# features_file_path = "GTZAN/just_bands_features_nyquist.csv"
scores_file_path = "GTZAN/original_song_scores.csv"


dataset = StandardizedFeaturesToScoreDataset(
    features_csv=features_file_path,
    scores_csv=scores_file_path,
    outliers_txt="GTZAN/outliers.txt",
    ignore_columns=[0,1],  # Ignore filename
)
dataset.analytical_normalization(feature_index=dataset.feature_columns.index("TotalRMS"))
merged_data = dataset.to_df()
merged_data = merged_data.drop(columns=["filename"]) # leave "label" for genre comparison
# Ensure score column is numeric (sometimes loaded as strings); coerce errors to NaN
if 'score' in merged_data.columns:
    merged_data['score'] = pd.to_numeric(merged_data['score'], errors='coerce')

letter = 'ρ' if METHOD == 'spearman' else 'r'

# rename E_x_y columns to x - y in merged_data and correlation_matrix
def rename_column(col_name):
    if col_name.startswith("E"):
        parts = col_name.split('_')
        if len(parts) == 3:
            return f"{parts[1]} Hz - {parts[2]} Hz"
    return col_name
merged_data.rename(columns=rename_column, inplace=True)

# get bin centers from column names
def get_bin_center(col_name):
    if ' - ' in col_name:
        parts = col_name.split(' - ')
        f_low = float(parts[0].replace(' Hz', ''))
        f_hi = float(parts[1].replace(' Hz', ''))
        return (f_low + f_hi) / 2.0
    return None
centers = []
for col in merged_data.columns[2:]: # skip 'label' and 'score'
    center = get_bin_center(col)
    if center is not None:
        centers.append(center)

# plot each feature as a bar chart with its correlation to score. Leave original ordering
correlations = merged_data.drop(columns=['label']).corr(method=METHOD)['score'].drop('score')

# Normalize the correlation values for color mapping
norm = plt.Normalize(correlations.min(), correlations.max())
colors = plt.cm.coolwarm(norm(correlations))

# normalize correlations within [-1, 1] (divide by max absolute value)
# NOTE: Maybe we can just do regular normalization instead of this? 
# e.g., (x - min) / (max - min). However, this way we preserve the sign of the correlation.
max_corr = max(np.abs(correlations.max()), np.abs(correlations.min()))
normalized_correlations = correlations / max_corr

# Let's do min-max normalization to [-1, 1]
# corr_min = correlations.min()
# corr_max = correlations.max()
# normalized_correlations = 2 * (correlations - corr_min) / (corr_max - corr_min) - 1

SIGMA = 0.5

# Create an audio equalizer (smoothed) from the normalized correlations
smoothed_eq = gaussian_filter1d(normalized_correlations.values, sigma=SIGMA)

print("Normalized correlations:")
print(normalized_correlations)

def apply_eq_with_curve(
    audio,
    sr,
    eq_curve,     # 1D array in [-1, 1], already smoothed
    centers,
    G_max_db=3.0,
    n_fft=2048,
    hop_length=None,
    drive=1.0,
):
    """
    Apply a static EQ curve (in [-1, 1]) to an audio signal using STFT.

    Parameters
    ----------
    audio : 1D numpy array
        Time-domain audio (mono).
    sr : int or float
        Sample rate of the audio.
    eq_curve : 1D numpy array
        Values in [-1, 1]. Length defines EQ frequency resolution from 0..sr/2.
    centers : 1D numpy array
        Frequencies (in Hz) corresponding to each bin in eq_curve.
    G_max_db : float
        Maximum absolute gain in dB. eq=-1 -> -G_max_db, +1 -> +G_max_db.
    n_fft : int, optional
        FFT size. If None, uses 2 * (len(eq_curve) - 1).
    hop_length : int, optional
        Hop size between STFT frames. If None, uses n_fft // 4.

    Returns
    -------
    audio_eq : 1D numpy array
        EQ'd audio, same length as input.
    """

    audio = np.asarray(audio, dtype=np.float64)

    # Choose n_fft so that positive-frequency bins roughly match eq_curve length
    if n_fft is None:
        n_fft = 2 * (len(eq_curve) - 1)  # so n_fft//2+1 == len(eq_curve)
    if hop_length is None:
        hop_length = n_fft // 4

    # STFT
    freqs, times, Zxx = stft(
        audio,
        fs=sr,
        nperseg=n_fft,
        noverlap=n_fft - hop_length,
        window="hann",
        boundary="zeros",
    )
    # freqs shape: (n_bins,), Zxx shape: (n_bins, n_frames)

    # Interpolate to STFT bin centers
    eq_resampled = np.interp(freqs, centers, eq_curve)

    # Convert [-1, 1] -> dB in [-G_max_db, +G_max_db]
    gains_db = eq_resampled * G_max_db
    gains_linear = 10.0 ** (gains_db / 20.0)  # shape (n_bins,)

    # Apply EQ: broadcast along time axis
    Zxx_eq = Zxx * gains_linear[:, np.newaxis]

    # Back to time domain
    _, audio_eq = istft(
        Zxx_eq,
        fs=sr,
        nperseg=n_fft,
        noverlap=n_fft - hop_length,
        window="hann",
        input_onesided=True,
    )

    # Trim/pad to original length
    if len(audio_eq) > len(audio):
        audio_eq = audio_eq[: len(audio)]
    elif len(audio_eq) < len(audio):
        pad = len(audio) - len(audio_eq)
        audio_eq = np.pad(audio_eq, (0, pad))
        
    def match_rms(audio, target_rms):
        rms = np.sqrt(np.mean(audio.astype(np.float64)**2))
        if rms > 0:
            k = target_rms / rms
            return audio * k
        else:
            return audio
    
    # Normalize to original audio RMS
    rms_original = np.sqrt(np.mean(audio.astype(np.float64)**2))
    audio_eq = match_rms(audio_eq, rms_original)
    audio_eq = np.tanh(drive * audio_eq / np.max(np.abs(audio_eq))) * np.max(np.abs(audio))  # soft clipping
    audio_eq = match_rms(audio_eq, rms_original)
    # clip to [-1, 1] if original audio is in that range
    if np.max(np.abs(audio)) <= 1.0:
        audio_eq = np.clip(audio_eq, -1.0, 1.0)
    rms_eq = np.sqrt(np.mean(audio_eq.astype(np.float64)**2))
    
    # convert back to frequencies for comparison to original
    new_fft = np.fft.rfft(audio_eq, n=n_fft)
    orig_fft = np.fft.rfft(audio, n=n_fft)
    # difference in dB
    diff_db = 20 * np.log10((np.abs(new_fft) + 1e-6) / (np.abs(orig_fft) + 1e-6))
    assert np.isclose(rms_original, rms_eq, atol=1e-6), "RMS normalization failed after EQ (original: {}, EQ'd: {})".format(rms_original, rms_eq)
    return audio_eq, diff_db

import librosa
def read_wave_file(file_path):
    y, sr = librosa.load(file_path, sr=22050, mono=True)
    return y, sr

if __name__ == "__main__":
    
    # compare normalized_correlations vs frequency distribution of each song genre
    genres = merged_data['label'].unique()
    genre_similarities = {}
    genre_good_frequency_percentages = {}
    genre_bad_frequency_percentages = {}
    genre_scores = {}
    genre_psds = {}
    for genre in genres:
        genre_data = merged_data[(merged_data['label'] == genre) & (merged_data['score'].notnull())]
        # compute similarity as the dot product between normalized_correlations and genre_data mean (excluding 'label' and 'score' columns)
        genre_mean = genre_data.drop(columns=['label', 'score']).mean().values
        # normalize genre_mean to match the range of normalized_correlations.values
        corr_min = normalized_correlations.values.min()
        corr_max = normalized_correlations.values.max()
        genre_min = genre_mean.min()
        genre_max = genre_mean.max()
        if genre_max > genre_min:
            genre_normalized = (genre_mean - genre_min) / (genre_max - genre_min) * (corr_max - corr_min) + corr_min
        else:
            genre_normalized = genre_mean
        similarity = np.dot(normalized_correlations.values, genre_normalized) / (np.linalg.norm(normalized_correlations.values) * np.linalg.norm(genre_normalized))
        # Compute energy in "good" vs "bad" frequency regions weighted by correlation strength
        # Good frequencies: where correlation is positive (should be boosted)
        # Bad frequencies: where correlation is negative (should be attenuated)
        positive_corr_mask = normalized_correlations.values >= 0
        negative_corr_mask = normalized_correlations.values < 0
        
        genre_psds[genre] = genre_normalized
        genre_normalized = (genre_mean - genre_min) / (genre_max - genre_min) # renormalize to [0, 1] for energy computation
        
        # Compute cosine similarities for good and bad frequency regions
        good_energy = np.dot(normalized_correlations.values[positive_corr_mask], genre_normalized[positive_corr_mask])

        bad_energy = np.dot(normalized_correlations.values[negative_corr_mask], genre_normalized[negative_corr_mask])
        
        # Normalize to percentages based on absolute contributions
        total_energy = np.abs(good_energy) + np.abs(bad_energy)
        if total_energy > 0:
            good_percentage = np.abs(good_energy) / total_energy
            bad_percentage = np.abs(bad_energy) / total_energy
        else:
            good_percentage = 0.5
            bad_percentage = 0.5
        
        genre_good_frequency_percentages[genre] = good_percentage
        genre_bad_frequency_percentages[genre] = bad_percentage
        genre_similarities[genre] = similarity
        genre_scores[genre] = genre_data['score'].mean().item()
    # sort genre_similarities by value
    genre_similarities = dict(sorted(genre_similarities.items(), key=lambda item: item[1], reverse=True))
    print("Genre similarities to Correlations curve:")
    for genre, sim in genre_similarities.items():
        print(f"{genre}: {sim}")
    
    # plot each
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(genre_similarities)), genre_similarities.values(), color='skyblue')
    # Add genre scores as points
    plt.xticks(range(len(genre_similarities)), genre_similarities.keys(), rotation=45, ha='right')
    plt.ylabel('Similarity to EQ Curve')
    plt.title('Genre Similarities to Computed EQ Curve')
    plt.tight_layout()
    plt.savefig('genre_similarity_to_correlations/genre_similarities_bar_chart.png')
    plt.show()
    
    # plot genre PSDs with normalized_correlations.values in separate subplots
    n_genres = len(genre_psds)
    n_cols = 3
    n_rows = (n_genres + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
    axes = axes.flatten() if n_genres > 1 else [axes]
    
    freq_axis = np.linspace(centers[0], centers[-1], len(normalized_correlations.values))
    
    for idx, (genre, psd) in enumerate(genre_psds.items()):
        ax = axes[idx]
        
        # Calculate difference from EQ curve
        difference = psd - normalized_correlations.values
        
        # Plot the difference filled area
        ax.fill_between(freq_axis, 0, difference, 
                        where=(difference >= 0), color='green', alpha=0.3, label='+', interpolate=True)
        ax.fill_between(freq_axis, 0, difference, 
                        where=(difference < 0), color='red', alpha=0.3, label='-', interpolate=True)
        
        # Plot the genre PSD and Correlations curve
        ax.plot(freq_axis, psd, label=f'{genre}', linewidth=2, color='blue')
        ax.plot(freq_axis, normalized_correlations.values, label='Correlations Curve', 
                linewidth=2, color='black', linestyle='--')
        ax.axhline(0, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)
        
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Normalized Value')
        ax.set_title(f'{genre} (Similarity: {genre_similarities[genre]:.3f})')
        # ax.legend(loc='best', fontsize='small')
        ax.grid(True, alpha=0.3)
    
    # Hide extra subplots
    for idx in range(n_genres, len(axes)):
        axes[idx].axis('off')
    
    # Create a single legend for all subplots
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.02), ncol=4, fontsize='medium')
    
    plt.suptitle('Genre PSDs vs Correlations Curve (Green=Above, Red=Below)', fontsize=14, y=1.00)
    plt.tight_layout()
    plt.savefig('genre_similarity_to_correlations/genre_psds_vs_correlations_curve.png', bbox_inches='tight')
    plt.show()
    
    # Create stacked bar chart for good and bad frequency sums
    plt.figure(figsize=(10, 6))
    genres_list = list(genre_good_frequency_percentages.keys())
    good_values = [genre_good_frequency_percentages[g] for g in genres_list]
    bad_values = [genre_bad_frequency_percentages[g] for g in genres_list]
    
    # Sort by good_values descending
    sorted_indices = np.argsort(good_values)[::-1]
    genres_list = [genres_list[i] for i in sorted_indices]
    good_values = [good_values[i] for i in sorted_indices]
    bad_values = [bad_values[i] for i in sorted_indices]
    
    x_pos = np.arange(len(genres_list))
    plt.bar(x_pos, good_values, color='red', label='Good Frequencies', alpha=0.8)
    plt.bar(x_pos, bad_values, bottom=good_values, color='blue', label='Bad Frequencies', alpha=0.8)
    
    plt.xticks(x_pos, genres_list, rotation=45, ha='right')
    plt.ylabel('Frequency Sum Proportion')
    plt.title('Good vs Bad Frequency Sums by Genre (Stacked)')
    plt.legend()
    plt.tight_layout()
    plt.savefig('genre_similarity_to_correlations/genre_good_bad_frequencies_stacked.png')
    plt.show()

    #export all figure data to CSV
    export_df = pd.DataFrame({'Frequency (Hz)': freq_axis, 'EQ Curve': normalized_correlations.values})
    for genre, psd in genre_psds.items():
        export_df[f'{genre} PSD'] = psd
    export_df.to_csv('genre_similarity_to_correlations/genre_psds_vs_correlations_curve.csv', index=False)
    
    # export genre similarities to CSV
    summary_df = pd.DataFrame({
        'Genre': list(genre_similarities.keys()),
        'Similarity': list(genre_similarities.values()),
    })
    summary_df.to_csv('genre_similarity_to_correlations/genre_similarities_summary.csv', index=False)
    
    # export good/bad frequency percentages to CSV
    gb_df = pd.DataFrame({
        'Genre': list(genre_good_frequency_percentages.keys()),
        'Good Frequency Percentage': list(genre_good_frequency_percentages.values()),
        'Bad Frequency Percentage': list(genre_bad_frequency_percentages.values()),
    })
    gb_df.to_csv('genre_similarity_to_correlations/genre_good_bad_frequency_percentages.csv', index=False)