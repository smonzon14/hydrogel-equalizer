# Hydrogel Equalizer

Derives an audio equalizer ("hydrogel EQ") from the correlation between per-band
spectral power density of songs and their measured hydrogel water-extraction
scores, applies that EQ to songs, and evaluates whether machine-learning models
(MLP, Ridge, Lasso, SVM) predict improved hydrogel scores for the EQ'd audio.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python GTZAN/download_dataset.py
```

The GTZAN audio dataset (~1.3 GB) is not committed to the repo.
`GTZAN/download_dataset.py` fetches it from
[Kaggle](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification)
(public, no account needed) and places the genre folders under
`GTZAN/genres_original/` (e.g. `GTZAN/genres_original/blues/blues.00000.wav`, ...).
It is a no-op if the audio is already there; pass `--force` to re-download.

All scripts are run from the repository root, e.g. `python hydrogel_eq.py` or
`python GTZAN/spearman_matrix.py`.

## Pipeline

1. **Feature extraction** — `GTZAN/process_features.py`
   Extracts per-band power-density deviation features (`EDevFrac_*`, global
   baseline) plus `TotalRMS` for every GTZAN song and writes
   `GTZAN/bands_dev_global_features_nyquist.csv`. This CSV is the input to all
   analysis and modeling below (a precomputed copy is committed).

2. **Statistical analysis**
   - `GTZAN/spearman_matrix.py` — Spearman correlation matrix and per-band
     correlation bar charts of band power density vs. water-loss score
     (writes the `frequency_band_correlation*.png` figures).
   - `plot_rms_vs_score.py` — Total RMS vs. score with 2nd-degree trend line;
     motivates the analytical (RMS-confound) normalization used everywhere else.

3. **Hydrogel EQ** — `hydrogel_eq.py`
   Builds the EQ curve from normalized band correlations (Gaussian-smoothed),
   applies it to an example song via STFT with RMS matching and soft clipping,
   and plots the EQ curve and before/after spectra. Exports
   `frequency_band_correlations_and_eq.csv` and `GTZAN/hydrogel_eq_compact.png`.

4. **Score models** (all use `GTZAN/standardized_features_to_score_dataset.py`
   for outlier removal, score normalization, and RMS-confound removal)
   - `GTZAN/score_estimator.py` — MLP score estimator (trained checkpoint at
     `GTZAN/models/score_estimator_bands_dev_nyquist.pth`).
   - `GTZAN/svm_estimator.py` — SVR baseline with grid search.
   - `GTZAN/evaluate_model.py` — evaluation of the trained MLP.

5. **EQ evaluation across models and gains**
   - `hydrogel_eq_test.py` — MLP-predicted score improvement on EQ'd songs.
   - `hydrogel_eq_ridge_test.py`, `hydrogel_eq_lasso_test.py`,
     `hydrogel_eq_svm_test.py` — same evaluation per model type.
   - `hydrogel_eq_all_models_comparison.py` — all models on one
     improvement-vs-gain plot (`hydrogel_eq_generated_songs/all_models_improvement_vs_gain.png`).
   - `hydrogel_eq_generate_all.py` — batch-applies the EQ at several gains and
     writes EQ'd WAVs + predicted scores to `hydrogel_eq_generated_songs/<gain>dB/`.

6. **Control EQs** (is the effect specific to the discovered bands?)
   - `hydrogel_eq_generate_controls.py` — writes inverted-EQ and random-EQ (band gains
     shuffled across bands) WAVs for the top-10 songs to
     `hydrogel_eq_generated_songs/controls_15dB/`, plus the curves and MLP-predicted
     scores (`control_eq_curves.*`, `control_eq_scores.csv`).
   - `hydrogel_eq_random_null.py` — predicted score of 100 random EQs vs. the optimal and
     inverted EQ for every model (`random_null_distribution.*`,
     `control_eqs_with_scores.png`).

7. **Genre analysis** — `hydrogel_eq_compare_genres.py`
   Compares each genre's average PSD shape against the score-correlation curve
   (similarity ranking, good/bad frequency breakdown). Outputs in
   `genre_similarity_to_correlations/`.

