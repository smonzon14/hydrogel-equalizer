import librosa
from glob import glob
import pandas as pd
import numpy as np, scipy.signal as sps, librosa
from scipy.signal import welch
from scipy import signal, fft
import time


def extract_features(file_path=None, y=None, sr=22050, num_mfcc=20, n_fft=2048, hop_length=512):
    """Extracts various audio features from a given audio file or waveform.
    Args:
        file_path (str): Path to the audio file.
        y (np.ndarray): Audio waveform. If provided, file_path is ignored.
        sr (int): Sample rate of the audio waveform.
        num_mfcc (int): Number of MFCCs to compute.
        n_fft (int): FFT window size.
        hop_length (int): Hop length for STFT.
    Returns:
        dict: A dictionary containing extracted features.
    """
    features: dict[str, float | str | None] = {
        "chroma_stft_mean": None, "chroma_stft_var": None,
        "rms_mean": None, "rms_var": None,
        "spectral_centroid_mean": None, "spectral_centroid_var": None,
        "spectral_bandwidth_mean": None, "spectral_bandwidth_var": None,
        "rolloff_mean": None, "rolloff_var": None,
        "zero_crossing_rate_mean": None, "zero_crossing_rate_var": None,
        "harmony_mean": None, "harmony_var": None,
        "perceptr_mean": None, "perceptr_var": None,
        "tempo": None,
        "mfcc1_mean": None, "mfcc1_var": None, "mfcc2_mean": None, "mfcc2_var": None,
        "mfcc3_mean": None, "mfcc3_var": None, "mfcc4_mean": None, "mfcc4_var": None,
        "mfcc5_mean": None, "mfcc5_var": None, "mfcc6_mean": None, "mfcc6_var": None,
        "mfcc7_mean": None, "mfcc7_var": None, "mfcc8_mean": None, "mfcc8_var": None,
        "mfcc9_mean": None, "mfcc9_var": None, "mfcc10_mean": None, "mfcc10_var": None,
        "mfcc11_mean": None, "mfcc11_var": None, "mfcc12_mean": None, "mfcc12_var": None,
        "mfcc13_mean": None, "mfcc13_var": None, "mfcc14_mean": None, "mfcc14_var": None,
        "mfcc15_mean": None, "mfcc15_var": None, "mfcc16_mean": None, "mfcc16_var": None,
        "mfcc17_mean": None, "mfcc17_var": None, "mfcc18_mean": None, "mfcc18_var": None,
        "mfcc19_mean": None, "mfcc19_var": None, "mfcc20_mean": None, "mfcc20_var": None,
    }
    try:
        if file_path is not None:
            y, sr = librosa.load(file_path, sr=sr)
    except Exception:
        return None
    
    assert y is not None, "Either file_path or waveform y must be provided."

    try:
        chromagram = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=512)
        features["chroma_stft_mean"] = chromagram.mean()
        features["chroma_stft_var"] = chromagram.var()

        RMSEn = librosa.feature.rms(y=y)
        features["rms_mean"] = RMSEn.mean()
        features["rms_var"] = RMSEn.var()

        spec_cent = librosa.feature.spectral_centroid(y=y)
        features["spectral_centroid_mean"] = spec_cent.mean()
        features["spectral_centroid_var"] = spec_cent.var()

        spec_band = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        features["spectral_bandwidth_mean"] = spec_band.mean()
        features["spectral_bandwidth_var"] = spec_band.var()

        spec_roll = librosa.feature.spectral_rolloff(y=y, sr=sr)
        features["rolloff_mean"] = spec_roll.mean()
        features["rolloff_var"] = spec_roll.var()

        zero_crossing = librosa.feature.zero_crossing_rate(y=y)
        features["zero_crossing_rate_mean"] = zero_crossing.mean()
        features["zero_crossing_rate_var"] = zero_crossing.var()

        harmony, perceptr = librosa.effects.hpss(y=y)
        features["harmony_mean"] = harmony.mean()
        features["harmony_var"] = harmony.var()
        features["perceptr_mean"] = perceptr.mean()
        features["perceptr_var"] = perceptr.var()

        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        features["tempo"] = tempo[0] # type: ignore

        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=num_mfcc, n_fft=n_fft, hop_length=hop_length)
        mfcc = mfcc.T

        for x in range(20):
            feat1 = f"mfcc{x+1}_mean"
            feat2 = f"mfcc{x+1}_var"
            features[feat1] = mfcc[:, x].mean()
            features[feat2] = mfcc[:, x].var()
    except Exception as e:
        # If any feature extraction fails, leave as None
        print("Error extracting features from audio file")
        print(e)
        pass

    return features

# -----------------------------
# Helpers
# -----------------------------
def bandpass(x, sr, f_lo, f_hi, order=6):
    hi = min(f_hi, 0.499*sr); lo = max(1.0, f_lo)
    if hi <= lo:
        return np.zeros_like(x)
    sos = sps.butter(order, [lo/(sr/2), hi/(sr/2)], btype='band', output='sos')
    return sps.sosfiltfilt(sos, x)

def band_rms(x, sr, f_lo, f_hi, order=6):
    xb = bandpass(x, sr, f_lo, f_hi, order)
    return float(np.sqrt(np.mean(xb**2) + 1e-12))

def safe_band(x, sr, f0, f1):
    hi = min(f1, 0.499*sr); lo = max(1.0, f0)
    if hi <= lo: return 0.0
    return band_rms(x, sr, lo, hi)

def crest_factor_db(x):
    peak = float(np.max(np.abs(x)) + 1e-9)
    rms  = float(np.sqrt(np.mean(x**2)) + 1e-12)
    return 20.0*np.log10(peak/rms)

def spectral_stationarity(x, sr):
    S = np.abs(librosa.stft(x, n_fft=2048, hop_length=512)) + 1e-9
    freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
    centroid = (S.T @ freqs) / S.sum(axis=0)
    return float(np.var(centroid))

# --- Envelope & modulation ---
def envelope(x, sr, lp_hz=20.0):
    env = np.abs(sps.hilbert(x))
    wc = min(lp_hz/(sr/2), 0.499)
    b, a = sps.butter(4, wc, btype='low')
    return sps.filtfilt(b, a, env)

def modulation_power_env(env, sr, f_lo, f_hi):
    E = env - env.mean()
    spec = np.abs(np.fft.rfft(E))
    freqs = np.fft.rfftfreq(len(E), 1.0/sr)
    sel = (freqs >= f_lo) & (freqs <= f_hi)
    if not np.any(sel):
        return 0.0
    return float(spec[sel].mean())

def emod_bands_fast(env, sr_env, bands, label="low"):
    """
    env: envelope (already low-passed)
    sr_env: sampling rate of envelope
    bands: list of (f_lo, f_hi) tuples
    Returns dict {f"Emod_{lo}_{hi}": mean_power_in_band}
    """
    E = env - env.mean()
    spec = np.abs(np.fft.rfft(E))
    freqs = np.fft.rfftfreq(len(E), d=1.0/sr_env)
    out = {}
    for (f_lo, f_hi) in bands:
        sel = (freqs >= f_lo) & (freqs <= f_hi)
        f_lo_str = str(f_lo).replace(".", "p")
        if not np.any(sel):
            out[f"Emod_{f_lo_str}_{f_hi}"] = 0.0
            continue
        out[f"Emod_{f_lo_str}_{f_hi}"] = float(spec[sel].mean())
    return out

def downsample_env(env, sr_env, target_sr=200):
    """Efficient, anti-aliased envelope downsample."""
    g = np.gcd(int(sr_env), int(target_sr))
    up, down = target_sr // g, sr_env // g
    return signal.resample_poly(env, up, down), float(target_sr)

# --- Flux & impulsiveness ---
def frame_rms(x, sr, win_ms=50, hop_ms=25):
    win = max(1, int(sr*win_ms/1000.0))
    hop = max(1, int(sr*hop_ms/1000.0))
    framed = librosa.util.frame(x, frame_length=win, hop_length=hop)
    r = np.sqrt((framed**2).mean(axis=0))
    return r, hop/sr

def rms_flux(x, sr):
    r, dt = frame_rms(x, sr)
    if len(r) < 3:
        return {"RMSFluxMean": 0.0}
    dr = np.diff(r)/max(dt, 1e-6)
    return {
        "RMSFluxMean": float(np.mean(np.abs(dr))),
    }

def tkeo_features(x):
    if len(x) < 3:
        return {"TKEOP95": 0.0}
    x1 = x[1:-1]
    psi = x1**2 - x[:-2]*x[2:]
    psi = np.abs(psi)
    return {"TKEOP95": float(np.percentile(psi, 95))}


# === ADDED: Spectral flatness in high band (1–10 kHz) ===
def spectral_flatness_band(x, sr, f_lo, f_hi, n_fft=2048, hop=512):
    S = np.abs(librosa.stft(x, n_fft=n_fft, hop_length=hop)) + 1e-12
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    mask = (freqs>=f_lo) & (freqs<=f_hi)
    if not np.any(mask):
        return 0.0
    Sb = S[mask]
    geomean = np.exp(np.mean(np.log(Sb), axis=0))
    arithmean = np.mean(Sb, axis=0) + 1e-12
    flat = geomean / arithmean
    return float(np.mean(flat))


def updated_extract_features(y, sr=22050.0):
    TotalRMS = np.sqrt(np.mean(y**2))

    # Gather linear RMS per band (use your safe_band or 10**log_rms)
    E20_150   = safe_band(y, sr, 20, 150)
    E150_250  = safe_band(y, sr, 150, 250)
    E1k_6k    = safe_band(y, sr, 1000, 6000)
    E8_10k    = safe_band(y, sr, 8000, 10000)

    # --- B) Spectral shape
    LowFrac      = float(E20_150/TotalRMS)
    HiFrac       = float(E8_10k/TotalRMS)
    LowVsMidALR = float(np.log((E20_150+1e-12)/(E1k_6k+1e-12)))

    # --- C) Modulation on low band
    y_low  = bandpass(y, sr, 20, 150)
    env_l  = envelope(y_low, sr, lp_hz=20.0)

    # Welch PSD of envelope (more stable than one FFT)
    f, Pxx = welch(env_l - env_l.mean(), fs=sr, nperseg=int(sr*4), noverlap=int(sr*2))
    df = f[1]-f[0] if len(f)>1 else 1.0
    sel_tot = (f>=0.2) & (f<=20.0)
    Ptot    = Pxx[sel_tot].sum()*df + 1e-12

    sel_12_16 = (f>=12.0) & (f<=16.0)
    PmodLow_12_16_abs = float(Pxx[sel_12_16].sum()*df)     # absolute modulation power
    Pslow_0p2_6        = float(Pxx[(f>=0.2)&(f<=6.0)].sum()*df + 1e-12)
    # PmodMixSlowVsFast  = float(np.log((PmodLow_12_16_abs+1e-12)/(Pslow_0p2_6)))

    # Mod depth (level-normalized envelope)
    e = env_l/(env_l.mean()+1e-12)
    # ModDepth_low = float(e.std())


    # --- E) Interaction
    LowPumpSynergy = float(LowFrac * PmodLow_12_16_abs)

    # --- Final compact dict
    feat = dict(
        LowFrac=LowFrac,
        HiFrac=HiFrac,
        LowVsMidALR=LowVsMidALR,
        PmodLow_12_16_abs=PmodLow_12_16_abs,
        LowPumpSynergy=LowPumpSynergy,
    )
    return feat

def extract_modulation_features(y, range_hz=(50,100), bands=[(0.5,2), (2,4), (4,8), (8,12), (12,16)], sr = 22050.0):
    """Extract modulation power features from the envelope of the audio signal.
    Args:
        y (np.ndarray): Audio waveform.
        range_hz (tuple): Frequency range for bandpass filtering before envelope extraction.
        bands (list): List of (f_lo, f_hi) tuples defining modulation frequency bands.
    Returns:
        dict: A dictionary containing modulation power features.
    """
    features = {}
    y_low  = bandpass(y, sr, range_hz[0], range_hz[1])
    env_l  = envelope(y_low, sr, lp_hz=20.0)    

    # Welch PSD of envelope (more stable than one FFT)
    f, Pxx = welch(env_l - env_l.mean(), fs=sr, nperseg=int(sr*4), noverlap=int(sr*2))
    for (f_lo, f_hi) in bands:
        sel = (f>=f_lo) & (f<=f_hi)
        if not np.any(sel):
            features[f"EmodLow_{f_lo}_{f_hi}"] = 0.0
            continue
        features[f"EModLow_{f_lo}_{f_hi}"] = float(Pxx[sel].sum()*(f[1]-f[0]))
    return features

# -----------------------------
low_bands = [(0,20), (20, 50), (50, 100), (100, 150), (150, 200), (200, 250), (250, 300), (300, 350), (350, 400)]
med_bands = [(400, 500), (500, 600), (600, 700), (700, 800), (800, 900), (900, 1000),
                (1000, 1200), (1200, 1400), (1400, 1600), (1600, 1800), (1800, 2000),
                (2000, 2500), (2500, 3000), (3000, 3500), (3500, 4000), (4000, 4500), (4500, 5000),
                (5000, 6000), (6000, 7000),]
high_bands = [(7000, 8000), (8000, 10000), (10000, 11025)]
# -----------------------------

def extract_just_bands(f=None,y=None, sr=22050.0, bands=None):
    if f is not None:
        try:
            y, sr = librosa.load(f, sr=sr)
        except Exception:
            return None
        if y is None:
            return None
    TotalRMS = np.sqrt(np.mean(y**2) + 1e-12)
    E_bands = {}
    ranges = np.concatenate([low_bands, med_bands, high_bands]) if bands is None else bands
    for (f_lo, f_hi) in ranges:
        E_bands[f"E_{f_lo}_{f_hi}"] = safe_band(y, sr, f_lo, f_hi)
    
    # for band in E_bands.keys():
    #     E_bands[band] = float(E_bands[band]**2 / TotalRMS**2)  # power fraction

    # g = np.prod([float(E_bands[b] + 1e-12) for b in E_bands.keys()])
    # geom_mean = g**(1.0/len(E_bands))
    # CLR = {}
    # for band in E_bands.keys():
    #     CLR[f"CLR_{band}"] = float(np.log((E_bands[band] + 1e-12)/ (geom_mean + 1e-12)))
    # E_bands.update(CLR)
    # # only_include = [
    # #     "E_50_100",
    # #     "E_150_200",
    # #     "E_400_500",
    # #     "E_1000_1200",
    # #     "E_1600_1800",
    # #     "E_2500_3000",
    # #     "E_3000_3500",
    # #     "E_3500_4000",
    # #     "E_4500_5000",
    # #     "E_6000_7000",
    # #     "E_7000_8000",
    # #     "E_8000_10000",
    # #     "E_10000_12000"
    # # ]
    # # E_bands = {k: E_bands[k] for k in only_include if k in E_bands.keys()}
    E_bands["TotalRMS"] = float(TotalRMS)
    return E_bands

def extract_bands_dev(
    f=None, y=None, sr=22050.0, bands=None,
    baseline="global",    # "power_law" | "global" | "ref"
    ref_density=None,        # list/array same len as bands if baseline=="ref"
    eps=1e-12
):
    """
    Returns per-band deviations (difference and fractional) while staying linear to power.
    Uses your `safe_band(y, sr, lo, hi)` (RMS amplitude in band).

    dev  = density - baseline_density           (signed difference, linear power/Hz)
    devf = (density / baseline_density) - 1     (signed fractional)
    """
    # load audio if needed
    if y is None:
        if f is None: return None
        try:
            y, sr = librosa.load(f, sr=sr, mono=True)
        except Exception:
            return None
    else:
        y = np.asarray(y, dtype=np.float64)
        if y.ndim > 1: y = y.mean(axis=-1)

    # bands
    if bands is None:
        try:
            ranges = np.concatenate([low_bands, med_bands, high_bands])
        except NameError:
            raise ValueError("Provide `bands` or define low_bands/med_bands/high_bands.")
    else:
        ranges = list(bands)

    # compute band power density via your helper (RMS^2 / bandwidth)
    centers = []
    widths  = []
    density = []
    for lo, hi in ranges:
        r = float(safe_band(y, sr, lo, hi))          # RMS amplitude in this band
        p = r * r                                    # linear power
        width = max(hi - lo, eps)
        density.append(p / width)                    # power per Hz
        widths.append(width)
        centers.append(0.5 * (lo + hi))

    density = np.asarray(density, dtype=np.float64)
    centers = np.asarray(centers, dtype=np.float64)
    widths  = np.asarray(widths,  dtype=np.float64)

    # choose baseline
    if baseline == "global":
        # one constant; weighted by bandwidth (fairer than plain mean)
        mu = np.average(density, weights=widths)
        base = np.full_like(density, mu)
    elif baseline == "ref":
        if ref_density is None or len(ref_density) != len(density):
            raise ValueError("Provide ref_density same length as bands for baseline='ref'.")
        base = np.asarray(ref_density, dtype=np.float64)
    elif baseline == "power_law":
        # fit log(density) ~ a + b*log(f)  (models 1/f^β tilt); skip DC/zeros
        mask = (centers > 0) & (density > 0)
        if mask.sum() >= 2:
            x = np.log(centers[mask])
            ylog = np.log(density[mask] + eps)
            b, a = np.polyfit(x, ylog, 1)           # slope b, intercept a
            base = np.exp(a + b * np.log(centers + eps))
        else:
            # fall back to global if not enough valid bands
            mu = np.average(density, weights=widths)
            base = np.full_like(density, mu)
    else:
        raise ValueError("baseline must be 'power_law', 'global', or 'ref'.")
    
    dev  = density - base
    devf = (density / (base + eps)) - 1.0

    # pack results
    out = {}
    for (lo, hi), d, df in zip(ranges, dev, devf):
        # out[f"EDev_{int(lo)}_{int(hi)}"]     = float(d)
        out[f"EDevFrac_{int(lo)}_{int(hi)}"] = float(df)

    # optional: diagnostics
    out["TotalRMS"] = float(np.sqrt(np.mean(y**2) + eps))
    
    # print("SUM DEVFRAC:", sum([out[f"EDevFrac_{int(lo)}_{int(hi)}"] for (lo, hi) in ranges]))
    # print("BASELINE TYPE:", baseline)
    # print("BASELINE VALUES:", base)
    # print("DENSITY VALUES:", density)
    # print("SUM DENSITY:", sum(density * widths))
    # print("DEVIATIONS:", dev)
    # print("FRACTIONAL DEVIATIONS:", devf)
    # print("Total RMS:", out["TotalRMS"])
    # import pdb; pdb.set_trace()
    # out["BaselineType"] = baseline
    return out

def extract_just_modulation(f=None,y=None, sr=22050.0, band=(50,200), mod_bands=[(0.5,2), (2,4), (4,8), (8,12), (12,16)]):
    if f is not None:
        try:
            y, sr = librosa.load(f, sr=sr)
        except Exception:
            return None
        if y is None:
            return None
        
    mod_features = {}
    pmod_feats = extract_modulation_features(y, range_hz=band, sr=sr)
    mod_features.update(pmod_feats)
    return mod_features

def extract_features_mixture(file_path=None, y=None, sr=22050):
    try:
        if file_path is not None:
            y, sr = librosa.load(file_path, sr=sr)
    except Exception:
        return None
    gtzan_features = extract_features(y=y, sr=sr)
    # physics_features = updated_extract_features(y, sr)
    band_features = extract_just_bands(y=y, sr=sr)
    mod_features = extract_just_modulation(y=y, sr=sr, band=(50,200))
    physics_features = {**band_features, **mod_features}
    return {**gtzan_features, **physics_features}


def parse_file_old_gtzan():
    num_segment=1
    num_mfcc=20
    sample_rate=22050
    n_fft=2048
    hop_length=512
    my_csv={"filename":[], "chroma_stft_mean": [], "chroma_stft_var": [], "rms_mean": [], "rms_var": [], "spectral_centroid_mean": [],
            "spectral_centroid_var": [], "spectral_bandwidth_mean": [], "spectral_bandwidth_var": [], "rolloff_mean": [], "rolloff_var": [],
            "zero_crossing_rate_mean": [], "zero_crossing_rate_var": [], "harmony_mean": [], "harmony_var": [], "perceptr_mean": [],
            "perceptr_var": [], "tempo": [], "mfcc1_mean": [], "mfcc1_var" : [], "mfcc2_mean" : [], "mfcc2_var" : [],
            "mfcc3_mean" : [], "mfcc3_var" : [], "mfcc4_mean" : [], "mfcc4_var" : [], "mfcc5_mean" : [], 
            "mfcc5_var" : [], "mfcc6_mean" : [], "mfcc6_var" : [], "mfcc7_mean" : [], "mfcc7_var" : [],
            "mfcc8_mean" : [], "mfcc8_var" : [], "mfcc9_mean" : [], "mfcc9_var" : [], "mfcc10_mean" : [], 
            "mfcc10_var" : [], "mfcc11_mean" : [], "mfcc11_var" : [], "mfcc12_mean" : [], "mfcc12_var" : [], 
            "mfcc13_mean" : [], "mfcc13_var" : [], "mfcc14_mean" : [], "mfcc14_var" : [], "mfcc15_mean" : [], 
            "mfcc15_var" : [], "mfcc16_mean" : [], "mfcc16_var" : [], "mfcc17_mean" : [], "mfcc17_var" : [], 
            "mfcc18_mean" : [], "mfcc18_var" : [], "mfcc19_mean" : [], "mfcc19_var" : [], "mfcc20_mean" : [], 
            "mfcc20_var":[], "label":[]}
    my_3_csv=my_csv.copy()

    # dataset_path="GTZAN/genres_original"
    dataset_path="old_ai"

    audio_files = glob(dataset_path + "/*/*")
    genre = glob(dataset_path + "/*")
    n_genres=len(genre)
    genre=[genre[x].split('/')[-1] for x in range(n_genres)]
    print(genre)

    genre=""
    for f in sorted(audio_files):
        if genre!=f.split('/')[-2]:
            genre=f.split('/')[-2]
            print("Processing " + genre + "...")
        fname=f.split('/')[-1]
        try:
            y, sr = librosa.load(f, sr=sample_rate)
        except:
            continue
        
        #Chromagram
        chroma_hop_length = 512 #5000?
        chromagram = librosa.feature.chroma_stft(y=y, sr=sample_rate, hop_length=chroma_hop_length)
        my_csv["chroma_stft_mean"].append(chromagram.mean())
        my_csv["chroma_stft_var"].append(chromagram.var())
        
        #Root Mean Square Energy
        RMSEn= librosa.feature.rms(y=y)
        my_csv["rms_mean"].append(RMSEn.mean())
        my_csv["rms_var"].append(RMSEn.var())
        
        #Spectral Centroid
        spec_cent=librosa.feature.spectral_centroid(y=y)
        my_csv["spectral_centroid_mean"].append(spec_cent.mean())
        my_csv["spectral_centroid_var"].append(spec_cent.var())
        
        #Spectral Bandwith
        spec_band=librosa.feature.spectral_bandwidth(y=y,sr=sample_rate)
        my_csv["spectral_bandwidth_mean"].append(spec_band.mean())
        my_csv["spectral_bandwidth_var"].append(spec_band.var())

        #Rolloff
        spec_roll=librosa.feature.spectral_rolloff(y=y,sr=sample_rate)
        my_csv["rolloff_mean"].append(spec_roll.mean())
        my_csv["rolloff_var"].append(spec_roll.var())
        
        #Zero Crossing Rate
        zero_crossing=librosa.feature.zero_crossing_rate(y=y)
        my_csv["zero_crossing_rate_mean"].append(zero_crossing.mean())
        my_csv["zero_crossing_rate_var"].append(zero_crossing.var())
        
        #Harmonics and Perceptrual 
        harmony, perceptr = librosa.effects.hpss(y=y)
        my_csv["harmony_mean"].append(harmony.mean())
        my_csv["harmony_var"].append(harmony.var())
        my_csv["perceptr_mean"].append(perceptr.mean())
        my_csv["perceptr_var"].append(perceptr.var())
        
        #Tempo
        tempo, _ = librosa.beat.beat_track(y=y, sr = sr)
        my_csv["tempo"].append(tempo[0]) # type: ignore

        mfcc=librosa.feature.mfcc(y=y,sr=sample_rate, n_mfcc=num_mfcc, n_fft=n_fft, hop_length=hop_length)
        mfcc=mfcc.T
        my_csv["filename"].append(fname)
        my_csv["label"].append(f.split('/')[-2])
        for x in range(20):
            feat1 = "mfcc" + str(x+1) + "_mean"
            feat2 = "mfcc" + str(x+1) + "_var"
            my_csv[feat1].append(mfcc[:,x].mean())
            my_csv[feat2].append(mfcc[:,x].var())
        print(fname)

    df = pd.DataFrame(my_csv)
    df.to_csv('old_ai.csv', index=False)


def extract_physics_features(file_path=None, y=None, sr=22050):
    """Extract physics-aware audio features for hydrogel actuation.
    Args:
        file_path (str): Path to the audio file.
        y (np.ndarray): Audio waveform. If provided, file_path is ignored.
        sr (int): Target sample rate for loading/resampling.
    Returns:
        dict: Feature dictionary.
    """
    import numpy as np
    import librosa
    import scipy.signal as sps
    from scipy.signal import welch
    from scipy import signal  # for resample_poly if used

    try:
        if file_path is not None:
            y, sr = librosa.load(file_path, sr=sr, mono=True)
    except Exception:
        return None

    assert y is not None, "Either file_path or waveform y must be provided."
    if len(y) < sr // 10:  # <100 ms
        return None

    

    # -----------------------------
    # Core bands (log-RMS), level controls
    TotalRMS = float(np.sqrt(np.mean(y**2)))

    
    # -----------------------------
    bands = {
        "E_20_150":  safe_band(y, sr, 20, 150),
        "E_150_250": safe_band(y, sr, 150, 250),
        "E_250_500": safe_band(y, sr, 250, 500),
        "E_500_1k":  safe_band(y, sr, 500, 1000),
        "E_1k_6k":   safe_band(y, sr, 1000, 6000),
        "E_6k_7k":   safe_band(y, sr, 6000, 7000),
        "E_7k_8k":   safe_band(y, sr, 7000, 8000),
        "E_8k_up":   safe_band(y, sr, 8000, 0.49*sr),  # upper limit
    }
    for k in list(bands.keys()):
        bands[k] = float(bands[k] / (1e-9 + TotalRMS))  # linear RMS denom

    # E_low150 = safe_band(y, sr, 20, 150)
    # LowFrac  = float(E_low150 / (1e-9 + TotalRMS))  # linear RMS denom

    # -----------------------------
    # Modulation spectrum on low band
    # -----------------------------
    y_low   = bandpass(y, sr, 20, 150)
    env_low = envelope(y_low, sr, lp_hz=20.0)
    env_med = envelope(bandpass(y, sr, 150, 8000), sr, lp_hz=20.0)
    env_high= envelope(bandpass(y, sr, 8000, 0.49*sr), sr, lp_hz=20.0)
    # If you want speed, you can downsample env_low to ~200 Hz and pass that sr into emod_bands_fast

    pmod_bands = [
        (0.2, 2), (2, 6), (6, 10), (10, 12),
        (12, 13), (13, 14), (14, 15), (15, 20),
    ]

    pmods_low = emod_bands_fast(env_low, sr, pmod_bands)
    pmod_low_total = 1e-12
    for k in pmods_low:
        pmod_low_total += pmods_low[k]
    for k in pmods_low:
        pmods_low[k] /= pmod_low_total

    pmods_med = emod_bands_fast(env_med, sr, pmod_bands, label="med")
    pmod_med_total = 1e-12
    for k in pmods_med:
        pmod_med_total += pmods_med[k]
    for k in pmods_med:
        pmods_med[k] /= pmod_med_total

    pmods_high= emod_bands_fast(env_high, sr, pmod_bands, label="high")
    pmod_high_total = 1e-12
    for k in pmods_high:
        pmod_high_total += pmods_high[k]
    for k in pmods_high:
        pmods_high[k] /= pmod_high_total

    # -----------------------------
    # Flux / impulse
    # -----------------------------
    flux_rms = rms_flux(y, sr)
    tkeo = tkeo_features(y)

    # Global crest (keep for interpretability)
    CrestDB = crest_factor_db(y)

    # Optional example (unused in final dict here):
    # Rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    # RolloffMean = float(Rolloff.mean())
    # RolloffVar  = float(Rolloff.var())

    # Perceptr = librosa.effects.hpss(y=y)
    # PerceptrMeanPerc = float(Perceptr[1].mean())
    # PerceptrVarPerc  = float(Perceptr[1].var())
    # PerceptrMeanHarm = float(Perceptr[0].mean())
    # PerceptrVarHarm  = float(Perceptr[0].var())

    # SpectralBandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    # SpectralBandwidthMean = float(SpectralBandwidth.mean())

    # Flatness_hi = spectral_flatness_band(y, sr, 1000, 10000)

    # -----------------------------
    # Assemble features
    # -----------------------------
    feat = {
        **bands,
        "TotalRMS": float(np.log10(1e-9 + TotalRMS)),
        # "LowFrac": LowFrac,

        # Modulation (low band)
        **pmods_low,
        **pmods_med,
        **pmods_high,

        # Flux / impulse
        **flux_rms,
        **tkeo,
        "Pmod_low_total": pmod_low_total,
        "Pmod_med_total": pmod_med_total,
        "Pmod_high_total": pmod_high_total,
        # Interpretable globals
        "CrestDB": CrestDB,
        # "Percptr_var_harm": PerceptrVarHarm,
    }

    return feat


def extract_only_rfe_features(file_path=None, y=None, sr=22050):
    try:
        if file_path is not None:
            y, sr = librosa.load(file_path, sr=sr)
    except Exception:
        return None
    # Model with alpha = 142.662:
    """
    Need to extract ONLY these features:

    E_50_100: 0.0279
    harmony_var: 0.0209
    mfcc5_mean: 0.0207
    perceptr_mean: -0.0200
    E_1000_1200: -0.0193
    mfcc13_mean: 0.0177
    EModLow_4_8: 0.0162
    mfcc12_mean: -0.0160
    mfcc3_var: -0.0158
    mfcc3_mean: 0.0135
    mfcc5_var: 0.0134
    mfcc11_var: 0.0131
    E_400_500: 0.0129
    harmony_mean: 0.0125
    rolloff_var: -0.0116
    E_150_200: 0.0115
    mfcc8_mean: -0.0111
    mfcc9_mean: -0.0110
    spectral_centroid_var: -0.0103
    E_100_150: 0.0102
    mfcc20_mean: 0.0101
    mfcc18_mean: -0.0098
    EModLow_2_4: 0.0097
    tempo: 0.0096
    mfcc4_mean: -0.0094
    perceptr_var: 0.0088
    E_3000_3500: 0.0086
    mfcc16_var: -0.0086
    mfcc15_mean: -0.0081
    rolloff_mean: -0.0081
    EModLow_12_16: 0.0075
    """
    feat = extract_features(file_path=None, y=y, sr=sr)
    band_feats = extract_just_bands(y=y, sr=sr, bands=[(50,100), (100,150), (150,200), (400,500), (1000,1200), (3000,3500)])
    mod_feats = extract_just_modulation(y=y, sr=sr, band=(50,200), mod_bands=[(2,4), (4,8), (12,16)])
    if feat is None or band_feats is None or mod_feats is None:
        raise ValueError("Feature extraction failed, returned None")
    feat.update(band_feats)
    feat.update(mod_feats)
    if feat is None:
        return None
    keys_to_keep = [
        "spectral_centroid_var",
        "rolloff_mean",
        "rolloff_var",
        "harmony_mean",
        "harmony_var",
        "perceptr_mean",
        "perceptr_var",
        "tempo",
        "mfcc3_mean",
        "mfcc3_var",
        "mfcc4_mean",
        "mfcc5_mean",
        "mfcc5_var",
        "mfcc8_mean",
        "mfcc9_mean",
        "mfcc11_var",
        "mfcc12_mean",
        "mfcc13_mean",
        "mfcc15_mean",
        "mfcc16_var",
        "mfcc18_mean",
        "mfcc20_mean",
        "E_50_100",
        "E_100_150",
        "E_150_200",
        "E_400_500",
        "E_1000_1200",
        "E_3000_3500",
        "EModLow_2_4",
        "EModLow_4_8",
        "EModLow_12_16"
    ]
    feat = {k: feat[k] for k in keys_to_keep if k in feat}
    return feat

if __name__ == "__main__":
    # dataset_path="old_ai"
    dataset_path="GTZAN/genres_original"
    
    audio_files = glob(dataset_path + "/*/*")
    genre = glob(dataset_path + "/*")
    n_genres=len(genre)
    genre=[genre[x].split('/')[-1] for x in range(n_genres)]
    print(genre)
    sample_rate=22050

    my_csv = None

    genre=""
    for f in sorted(audio_files):
        if genre!=f.split('/')[-2]:
            genre=f.split('/')[-2]
            print("Processing " + genre + "...")
        fname=f.split('/')[-1]
        features = extract_bands_dev(f, baseline="global")
        if features is None:
            continue

        if my_csv is None:
            my_csv = {"filename":[], "label":[]}
            for k in features.keys():
                my_csv[k] = []
        for k in features:
            my_csv[k].append(features[k])
        my_csv["filename"].append(fname)
        my_csv["label"].append(f.split('/')[-2])
        print(fname)
        for k in my_csv:
            if len(my_csv[k]) != len(my_csv["filename"]):
                print(f"Length mismatch in {k}: {len(my_csv[k])} vs {len(my_csv['filename'])}")
    df = pd.DataFrame(my_csv)
    df.to_csv('GTZAN/bands_dev_global_features_nyquist.csv', index=False)
    