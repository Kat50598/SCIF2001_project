import json
import logging
import os
import pdb
from pathlib import Path

import mne
import numpy as np
import pandas as pd
import scipy
from scipy import signal
from scipy.integrate import simpson

bands = {
    "delta": (0.5, 4),
    "theta": (4, 8),
    "alpha": (8, 12),
    "sigma": (12, 15),
    "beta": (15, 30),
}


def get_band_power(fft_freqs, psd, band):
    low_freq, high_freq = bands[band]
    freq_res = fft_freqs[1] - fft_freqs[0]
    band_indices = np.where((fft_freqs >= low_freq) & (fft_freqs <= high_freq))[0]
    power = np.sum(psd[band_indices]) * freq_res
    return power


def feature_extract_fft(raw_cleaned_df, epoch, channel):
    epoch_filt = raw_cleaned_df["epoch"] == epoch
    s = raw_cleaned_df[epoch_filt][channel]
    t = raw_cleaned_df[epoch_filt]["time"]

    label = raw_cleaned_df[epoch_filt]["condition"].unique()[0]

    d = 30 / (len(t) - 1)
    n = len(t)

    fft = 2 * np.fft.fft(s) / n
    fft_freqs = np.fft.fftfreq(n, d)

    intensity = np.abs(fft) ** 2

    peak_freq = np.abs(fft_freqs[np.argmax(intensity)])
    total_power = intensity.sum()
    delta_power = get_band_power(fft_freqs, intensity, "delta")
    theta_power = get_band_power(fft_freqs, intensity, "theta")
    alpha_power = get_band_power(fft_freqs, intensity, "alpha")
    sigma_power = get_band_power(fft_freqs, intensity, "sigma")
    beta_power = get_band_power(fft_freqs, intensity, "beta")

    return {
        "peak_freq": peak_freq,
        "total_power": total_power,
        "delta_power": delta_power,
        "theta_power": theta_power,
        "alpha_power": alpha_power,
        "sigma_power": sigma_power,
        "beta_power": beta_power,
    }, label


def feature_extract_welch(raw_cleaned_df, epoch, channel):
    epoch_filt = raw_cleaned_df["epoch"] == epoch
    s = raw_cleaned_df[epoch_filt][channel].to_numpy()
    t = raw_cleaned_df[epoch_filt]["time"]

    fs = 256
    nperseg = 2 * fs
    # The 'welch' function returns frequencies and power spectral density
    fft_freqs, intensity = signal.welch(s, fs=fs, nperseg=nperseg)

    label = raw_cleaned_df[epoch_filt]["condition"].unique()[0]

    peak_freq = np.abs(fft_freqs[np.argmax(intensity)])
    total_power = intensity.sum()
    delta_power = get_band_power(fft_freqs, intensity, "delta")
    theta_power = get_band_power(fft_freqs, intensity, "theta")
    alpha_power = get_band_power(fft_freqs, intensity, "alpha")
    sigma_power = get_band_power(fft_freqs, intensity, "sigma")
    beta_power = get_band_power(fft_freqs, intensity, "beta")

    return {
        "peak_freq": peak_freq,
        "total_power": total_power,
        "delta_power": delta_power,
        "theta_power": theta_power,
        "alpha_power": alpha_power,
        "sigma_power": sigma_power,
        "beta_power": beta_power,
    }, label


def data_features_frame(raw_cleaned_df, output_dir, sub):
    """
    Geneates dataframe with extracted features for one patient and saves it in
    extracted_features folder
    Takes
    - channels - a list of EEG channel labels to extract for a given subject
    - raw_cleaned_df: the raw cleaned data file for a single patient
    - subject number as a string

    """
    channels_to_keep = ["C3:A2", "C4:A1", "O2:A1", "F4:A1", "O1:A2", "F3:A2"]

    all_epoch_features = []

    for epoch in raw_cleaned_df["epoch"].unique():
        features_for_this_epoch = {}
        label = None
        for ch in channels_to_keep:
            features, label = feature_extract_welch(raw_cleaned_df, epoch, ch)
            for feature_name, value in features.items():
                features_for_this_epoch[f"{ch}_{feature_name}"] = value
        features_for_this_epoch["label"] = label

        all_epoch_features.append(features_for_this_epoch)

    # saves the extracted feature data frame to output directory
    final_df = pd.DataFrame(all_epoch_features)
    out_path = os.path.join(output_dir, f"{sub}_extracted_features")
    os.makedirs(output_dir, exist_ok=True)
    final_df.to_csv(out_path, index=False)


def main():

    data_path = Path(__file__).parent.parent / "EEG_data"
    dirs = os.listdir(data_path / "ds005207")
    subs = [d for d in dirs if d.startswith("sub")]

    output_dir = data_path / "extracted_features"

    for sub in subs:
        data_path = Path("../") / "EEG_data" / "cleaned_data"
        sample = data_path / f"{sub}_cleaned-epo.fif"

        cleaned_data_file = sample
        epochs = mne.read_epochs(cleaned_data_file, preload=True, verbose=False)
        raw_cleaned_df = epochs.to_data_frame()

        data_features_frame(raw_cleaned_df, output_dir)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    main()
