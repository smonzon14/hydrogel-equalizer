import pandas as pd
import numpy as np
# import seaborn as sns  # unused
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
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
merged_data = merged_data.drop(columns=["label", "filename"])

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
for col in merged_data.columns[1:]:
    center = get_bin_center(col)
    if center is not None:
        centers.append(center)

# plot each feature as a bar chart with its correlation to score. Leave original ordering
correlations = merged_data.corr(method=METHOD)['score'].drop('score')

# Normalize the correlation values for color mapping
norm = plt.Normalize(correlations.min(), correlations.max())
# Use tab:blue for positive correlations, red for negative
colors = ['tab:blue' if corr >= 0 else 'tab:red' for corr in correlations]

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
        # clipping lowers the RMS slightly, so re-match and re-clip until both hold
        for _ in range(50):
            audio_eq = np.clip(audio_eq, -1.0, 1.0)
            if np.isclose(np.sqrt(np.mean(audio_eq**2)), rms_original, atol=1e-6):
                break
            audio_eq = match_rms(audio_eq, rms_original)
        else:
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
    GAIN = 15  # dB
    SIGMA = 0.5  # for Gaussian smoothing
    DRIVE = 3.1  # soft clipping drive
    smoothed_eq = gaussian_filter1d(normalized_correlations.values, sigma=SIGMA)
    song_example = "hiphop.00037.wav"
    # display the smoothed equalizer
    fig, ax1 = plt.subplots(figsize=(10, 6))
    audio, sr = read_wave_file(f"GTZAN/genres_original/hiphop/{song_example}")  # Replace with your WAV file path
    
    # smoothed_eq = np.concatenate(([0], smoothed_eq, [smoothed_eq[-1]]))
    # centers = np.concatenate(([0], centers, [sr / 2]))
    # Plot smoothed_eq on left y-axis
    ax1.set_xlabel(r"Frequency Band (${f_{low}}$ - ${f_{hi}}$)")
    ax1.set_ylabel("Equalizer Gain (dB)")
    color = 'tab:blue'
    ax1.plot(merged_data.columns[1:], smoothed_eq * GAIN, color=color, label=f'Smoothed Equalizer (σ={SIGMA})')
    ax1.set_ylim((smoothed_eq*GAIN).min() - 1, (smoothed_eq*GAIN).max() + 1)
    ax1.tick_params(axis='y')

    # Plot correlations on right y-axis
    ax2 = ax1.twinx()
    ax2.set_ylabel(f"Correlation to Score ({letter})")
    # Calculate alpha values based on magnitude of correlations
    corr_abs = np.abs(correlations)
    alphas = (corr_abs - corr_abs.min()) / (corr_abs.max() - corr_abs.min())
    alphas = 0.3 + 0.7 * alphas  # Scale to [0.3, 1.0] range for visibility
    for i, (col, corr, color, alpha) in enumerate(zip(merged_data.columns[1:], correlations, colors, alphas)):
        ax2.bar(col, corr, color=color, alpha=alpha)
    # Add dummy bar for legend
    ax2.bar([], [], color='gray', alpha=0.6, label='Correlation')
    ax2.tick_params(axis='y')
    ax2.set_ylim(correlations.min() - 0.1, correlations.max() + 0.1)

    # save CSV of columns, correlations, and smoothed_eq
    eq_df = pd.DataFrame({
        'Frequency Band': merged_data.columns[1:],
        'Correlation': correlations.values,
        'Smoothed EQ (normalized)': smoothed_eq,
        f'Smoothed EQ ({GAIN} dB)': smoothed_eq * GAIN,
    })
    eq_df.to_csv("frequency_band_correlations_and_eq.csv", index=False)
    
    fig.suptitle('Audio Equalizer (Gain={} dB) & Frequency Band Correlations'.format(GAIN))
    fig.autofmt_xdate(rotation=45)
    plt.grid()
    plt.tight_layout()
    # Combine legends from both axes
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc='lower right')
    plt.xticks(rotation=45, ha='right')
    
    # plot eq and correaltions in their own mini plots below
    
    plt.figure(figsize=(7, 3))
    plt.subplot(1, 2, 1)
    
    # Calculate alpha values based on magnitude of correlations
    corr_abs = np.abs(correlations)
    alphas = (corr_abs - corr_abs.min()) / (corr_abs.max() - corr_abs.min())
    alphas = 0.3 + 0.7 * alphas  # Scale to [0.3, 1.0] range for visibility
    for i, (col, corr, color, alpha) in enumerate(zip(merged_data.columns[1:], correlations, colors, alphas)):
        plt.bar(col, corr, color=color, alpha=alpha)
    plt.xlabel(r"Frequency (Hz)")
    plt.ylabel(f"Correlation to Score ({letter})")
    # plt.title('Frequency Band Correlations')
    plt.axhline(0, color='gray', linestyle='--', linewidth=1)  # Only one horizontal grid line at y=0
    # plt.grid(axis='x')
    plt.xticks([])
    # plt.xticks([0, len(merged_data.columns[1:]) - 1], ["0", "11,025"], ha='right')
    # plt.grid()
    
    
    plt.subplot(1, 2, 2)
    plt.plot(merged_data.columns[1:], smoothed_eq * GAIN, label=f'Smoothed Equalizer (σ={SIGMA})')
    # Fill above and below y=0 for visual difference
    # Convert numpy boolean arrays to Python lists for 'where' argument
    above_zero = (smoothed_eq * GAIN >= 0).tolist()
    below_zero = (smoothed_eq * GAIN < 0).tolist()

    # Fill above and below y=0, handling zero crossings for better visual fill
    x = np.arange(len(smoothed_eq))
    y = smoothed_eq * GAIN
    for i in range(len(y) - 1):
        x0, x1 = x[i], x[i+1]
        y0, y1 = y[i], y[i+1]
        # Check for zero crossing
        if (y0 >= 0 and y1 < 0) or (y0 < 0 and y1 >= 0):
            # Linear interpolation to find crossing
            x_cross = x0 + (0 - y0) * (x1 - x0) / (y1 - y0)
            # Fill up to the crossing
            if y0 >= 0:
                plt.fill_between([x0, x_cross], [y0, 0], 0, color='gray', alpha=0.2)
                plt.fill_between([x_cross, x1], [0, y1], 0, color='gray', alpha=0.2)
            else:
                plt.fill_between([x0, x_cross], [y0, 0], 0, color='gray', alpha=0.2)
                plt.fill_between([x_cross, x1], [0, y1], 0, color='gray', alpha=0.2)
        else:
            # No crossing, fill once per segment
            color = 'gray' if y0 >= 0 and y1 >= 0 else 'gray'
            plt.fill_between([x0, x1], [y0, y1], 0, color=color, alpha=0.1)

    plt.xlabel(r"Frequency (Hz)")
    plt.ylabel("Equalizer Gain (dB)")
    # plt.title('Smoothed Equalizer Curve')
    # plt.xticks([0, len(merged_data.columns[1:]) - 1], ["0", "11,025"], ha='right')
    plt.axhline(0, color='gray', linestyle='--', linewidth=1)  # Only one horizontal grid line at y=0
    # plt.grid(axis='x')
    plt.xticks([])
    # Read an example WAV file
    plt.tight_layout()

    # Compact resulting EQ curve (companion to GTZAN/frequency_band_correlations_compact.png):
    # the hydrogel equalizer derived from the correlations -> normalized -> smoothed -> dB.
    BOOST_BLUE = "#2878B8"
    CUT_RED = "#B43426"
    eq_db = smoothed_eq * GAIN
    x_eq = np.arange(len(eq_db))
    fig_eq, ax_eq = plt.subplots(figsize=(3.9, 2.5))
    ax_eq.plot(x_eq, eq_db, color="black", linewidth=1.2)
    ax_eq.fill_between(x_eq, eq_db, 0, where=(eq_db >= 0), interpolate=True,
                       color=BOOST_BLUE, alpha=0.45)
    ax_eq.fill_between(x_eq, eq_db, 0, where=(eq_db < 0), interpolate=True,
                       color=CUT_RED, alpha=0.45)
    ax_eq.axhline(0, color='gray', linestyle='--', linewidth=1)
    ax_eq.set_xlabel(r"Frequency (Hz)")
    ax_eq.set_ylabel("Equalizer Gain (dB)")
    ax_eq.set_xticks([])
    ax_eq.yaxis.set_major_locator(MaxNLocator(nbins=4))
    fig_eq.tight_layout()
    fig_eq.savefig("GTZAN/hydrogel_eq_compact.png", dpi=300)

    plt.show()
    
    # Apply EQ
    audio_eq, diff_db = apply_eq_with_curve(
        audio,
        sr,
        eq_curve=smoothed_eq,
        centers=np.array(centers),
        G_max_db=GAIN,
        drive=DRIVE,
    )

    # save EQ'd audio to WAV file
    import soundfile as sf
    sf.write("example_eq.wav", audio_eq, sr)

    # Plot original and EQ'd audio frequency vs amplitude overlayed by smoothed_eq
    n_fft = 4096
    original_fft = np.fft.rfft(audio, n=n_fft)
    eq_fft = np.fft.rfft(audio_eq, n=n_fft)

    # For plotting, use index positions for centers so that each center is spaced equally
    center_indices = np.arange(len(centers))

    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Interpolate FFT magnitudes to match the number of centers for fair comparison
    freqs = np.fft.rfftfreq(n_fft, d=1/sr)
    original_magnitudes = np.abs(original_fft)
    eq_magnitudes = np.abs(eq_fft)
    #plot n_fft/2+1 points
    # Interpolate to get more points between true centers for smoother curves
    interp_points = 9  # Number of points between each pair of centers
    dense_centers = []
    dense_indices = []
    for i in range(len(centers) - 1):
        # Interpolate between each pair of centers
        segment = np.linspace(centers[i], centers[i+1], interp_points + 1, endpoint=False)
        dense_centers.extend(segment)
        index_segment = np.linspace(center_indices[i], center_indices[i+1], interp_points + 1, endpoint=False)
        dense_indices.extend(index_segment)
    # Add the last center and index
    dense_centers.append(centers[-1])
    dense_indices.append(center_indices[-1])
    dense_centers = np.array(dense_centers)
    dense_indices = np.array(dense_indices)

    original_magnitudes_interp = np.interp(dense_centers, freqs, original_magnitudes)
    eq_magnitudes_interp = np.interp(dense_centers, freqs, eq_magnitudes)
    # Plot smoothed_eq on its own right y-axis (dB scale)
    eq_alpha = 0.2
    # ax1.plot(center_indices, smoothed_eq, label='Smoothed EQ Curve', color='black', alpha=eq_alpha)
    # ax1.set_ylabel('EQ Gain (dB)', color='black')
    # # horizontal line at y=0
    # ax1.axhline(0, color='gray', linestyle='--', linewidth=0.8)
    # # fill between smoothed_eq and y=0
    # # Fill between points, including between each pair of points for smooth color transitions
    # # Fill between the whole curve and y=0: green above, red below
    # ax1.fill_between(center_indices, smoothed_eq, 0, where=smoothed_eq >= 0, color='blue', alpha=eq_alpha)
    # ax1.fill_between(center_indices, smoothed_eq, 0, where=smoothed_eq < 0, color='red', alpha=eq_alpha)
    # for i in range(len(center_indices) - 1):
    #     x = [center_indices[i], center_indices[i+1]]
    #     y = [smoothed_eq[i], smoothed_eq[i+1]]
    #     if y[0] >= 0 and y[1] >= 0:
    #         pass
    #     elif y[0] < 0 and y[1] < 0:
    #         pass
    #     else:
    #         # If sign changes between points, fill each triangle separately for correct color
    #         # Find zero crossing
    #         x0, x1 = x
    #         y0, y1 = y
    #         x_cross = x0 + (0 - y0) * (x1 - x0) / (y1 - y0)
    #         if y0 > 0:
    #             ax1.fill_between([x0, x_cross], [y0, 0], 0, color='blue', alpha=eq_alpha)
    #             ax1.fill_between([x_cross, x1], [0, y1], 0, color='red', alpha=eq_alpha)
    #         else:
    #             ax1.fill_between([x0, x_cross], [y0, 0], 0, color='red', alpha=eq_alpha)
    #             ax1.fill_between([x_cross, x1], [0, y1], 0, color='blue', alpha=eq_alpha)



    ax1.plot(dense_indices, 20 * np.log10(original_magnitudes_interp + 1e-6), label='Original Audio', alpha=1.0)
    ax1.plot(dense_indices, 20 * np.log10(eq_magnitudes_interp + 1e-6), label='EQ\'d Audio', alpha=1.0)
    # color difference area between original and eq'd
    ax1.fill_between(dense_indices, 20 * np.log10(original_magnitudes_interp + 1e-6), 20 * np.log10(eq_magnitudes_interp + 1e-6), where=(eq_magnitudes_interp >= original_magnitudes_interp), color='blue', alpha=0.3, label='Increase')
    ax1.fill_between(dense_indices, 20 * np.log10(original_magnitudes_interp + 1e-6), 20 * np.log10(eq_magnitudes_interp + 1e-6), where=(eq_magnitudes_interp < original_magnitudes_interp), color='red', alpha=0.3, label='Decrease')
    ax1.set_xlabel(r"Frequency Band (${f_{low}}$ - ${f_{hi}}$)")
    ax1.set_ylabel('Amplitude (dB)')
    ax1.set_title('Frequency Response Before and After EQ for {} (Gain={} dB, Drive={})'.format(song_example,GAIN, DRIVE))
    ax1.legend(loc='upper right')


    # ax3 = ax1.twinx()
    # ax3.spines['right'].set_position(('outward', 60))  # Offset the third y-axis
    # diff_freqs = np.fft.rfftfreq(len(diff_db)*2 - 2, d=1/sr)
    # interp = np.interp(dense_centers, diff_freqs, diff_db)
    # diff_db = interp
    # ax3.plot(dense_indices, diff_db, label='EQ Applied (dB)', color='green', alpha=0.7)
    # ax3.set_ylabel('EQ Applied (dB)', color='green')
    # Set x-ticks to be at each center index, with labels as the frequency (rounded for readability)
    ax1.set_xticks(center_indices)
    ax1.set_xticklabels(merged_data.columns[1:])

    # add line at y=0 for diff_db
    # ax3.axhline(0, color='gray', linestyle='--', linewidth=0.8)

    plt.setp(ax1.get_xticklabels(), ha='right')
    fig.autofmt_xdate(rotation=45)

    # Remove plt.xticks here, as rotation is already set above for ax3
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = [],[] #ax2.get_legend_handles_labels()
    lines3, labels3 = [],[] #ax3.get_legend_handles_labels()

    ax1.legend(lines1 + lines2 + lines3, labels1 + labels2 + labels3, loc='upper right')

    plt.tight_layout()
    plt.show()