import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
import mne
from scipy.signal import butter, sosfilt
from sklearn.cross_decomposition import CCA
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

# --- Constants ---
SFREQ = 250.0
MI_CHANNELS = ["C3", "CZ", "C4"]
SSVEP_CHANNELS_ALL = ["FZ", "C3", "CZ", "C4", "PZ", "P07", "OZ", "P08"]
SSVEP_CHANNELS_CCA = ["PZ", "P07", "OZ", "P08", "C3", "C4"]
STIM_FREQS = {"Left": 10.0, "Right": 13.0, "Forward": 7.0, "Backward": 8.0}
LABELS_SSVEP = list(STIM_FREQS.keys())

# --- Dataset Loading Classes ---

class EEG_Dataset(Dataset):
    """
    Base class for loading and handling EEG data from the competition format.
    """
    def __init__(self, index_csv: str, base_path: str, ch_names: list, task: str, transform=None):
        self.base_path = base_path
        self.ch_names = ch_names
        self.task = task
        self.transform = transform
        
        # Load the index CSV and filter for the specific task
        full_df = pd.read_csv(os.path.join(base_path, index_csv))
        self.df = full_df[full_df["task"] == self.task].reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Determine split (train, validation, or test)
        id_num = row["id"]
        if "train" in row['eeg_path']:
            split = "train"
        elif "validation" in row['eeg_path']:
            split = "validation"
        else:
            split = "test"
            
        # Path to session CSV
        eeg_path = os.path.join(self.base_path, row["eeg_path"])
        
        # Load full session
        try:
            sess_df = pd.read_csv(eeg_path)
        except Exception as e:
            print(f"Error loading {eeg_path}: {e}")
            return None, None

        # Determine samples per trial
        samp = (9 if self.task == "MI" else 7) * SFREQ
        n = int(samp)
        start = (int(row["trial"]) - 1) * n
        end = start + n

        # Slice out EEG channels
        data = sess_df[self.ch_names].iloc[start:end].to_numpy().T
        
        # Apply MNE transform if provided
        if self.transform:
            with mne.utils.use_log_level("WARNING"):
                info = mne.create_info(ch_names=self.ch_names, sfreq=SFREQ, ch_types=["eeg"] * len(self.ch_names))
                raw = mne.io.RawArray(data, info)
                raw = self.transform(raw)
                data = raw.get_data()

        eeg_tensor = torch.from_numpy(data).float()

        if "label" in self.df.columns:
            label = row["label"]
            return eeg_tensor, label
        else:
            return eeg_tensor

# --- Feature Extraction Functions ---

def extract_mi_features(eeg_tensor):
    """
    Extracts combined PSD and FFT features for a single MI trial.
    """
    # 1. PSD Features
    info = mne.create_info(ch_names=MI_CHANNELS, sfreq=SFREQ, ch_types=["eeg"] * len(MI_CHANNELS))
    raw = mne.io.RawArray(eeg_tensor.numpy(), info, verbose=False)
    psds, freqs = raw.compute_psd(method="welch", fmin=8.0, fmax=30.0, n_fft=1024, verbose=False).get_data(return_freqs=True)
    
    freqs_of_interest = np.arange(8, 31, 1)
    psd_feature_vector = []
    for freq in freqs_of_interest:
        idx = np.argmin(np.abs(freqs - freq))
        psd_feature_vector.extend(psds[:, idx])

    # 2. FFT Features
    x = eeg_tensor.numpy().T * np.hamming(2250)[:, None]
    fft_vals = np.fft.rfft(x, axis=0)
    fft_freqs = np.fft.rfftfreq(2250, d=1/SFREQ)
    mask = (fft_freqs >= 8) & (fft_freqs <= 30)
    spectrum = fft_vals[mask, :]
    fft_feature_vector = spectrum.flatten().real

    return np.concatenate((psd_feature_vector, fft_feature_vector))

def extract_ssvep_features(eeg_tensor):
    """
    Extracts combined FBCCA and PSD features for a single SSVEP trial.
    """
    # 1. FBCCA Features
    fbcca_scores = _get_fbcca_scores(eeg_tensor)
    fbcca_feature_vector = np.array([fbcca_scores[label] for label in sorted(STIM_FREQS.keys())]).flatten()

    # 2. PSD Features
    info = mne.create_info(ch_names=SSVEP_CHANNELS_ALL, sfreq=SFREQ, ch_types=["eeg"] * len(SSVEP_CHANNELS_ALL))
    raw = mne.io.RawArray(eeg_tensor.numpy(), info, verbose=False)
    psds, freqs = raw.compute_psd(method="welch", fmin=1.0, fmax=60.0, n_fft=1024, verbose=False).get_data(return_freqs=True)
    
    freqs_of_interest = []
    for freq in STIM_FREQS.values():
        freqs_of_interest.append(freq)
        freqs_of_interest.append(freq * 2) # 2nd harmonic
        
    psd_feature_vector = []
    for freq in freqs_of_interest:
        idx = np.argmin(np.abs(freqs - freq))
        psd_feature_vector.extend(psds[:, idx])
        
    return np.concatenate((fbcca_feature_vector, psd_feature_vector))

# --- Helper functions for SSVEP ---

def _generate_reference_signals(freq, n_samples, sfreq, n_harmonics=2):
    t = np.arange(n_samples) / sfreq
    refs = []
    for h in range(1, n_harmonics + 1):
        refs.append(np.sin(2 * np.pi * h * freq * t))
        refs.append(np.cos(2 * np.pi * h * freq * t))
    return np.stack(refs).T

def _create_filter_bank(sfreq, n_bands=5, band_width=8, start_freq=6.0):
    filters = []
    for i in range(n_bands):
        low_cut = start_freq + i * band_width
        high_cut = start_freq + (i + 1) * band_width
        if high_cut >= sfreq / 2:
            high_cut = sfreq / 2 - 0.1
        sos = butter(5, [low_cut, high_cut], btype="band", fs=sfreq, output="sos")
        filters.append(sos)
        if high_cut >= sfreq / 2 - 0.1:
            break
    return filters

def _get_fbcca_scores(eeg_tensor, n_harmonics=2, n_bands=5, band_width=8, start_freq=6.0):
    ssvep_ch_indices = [SSVEP_CHANNELS_ALL.index(ch) for ch in SSVEP_CHANNELS_CCA]
    eeg_data = eeg_tensor.squeeze(0).numpy()[ssvep_ch_indices, :]
    
    trim_samples = int(0.5 * SFREQ)
    eeg_data = eeg_data[:, trim_samples:-trim_samples]
    n_samples = eeg_data.shape[1]

    filter_bank = _create_filter_bank(SFREQ, n_bands, band_width, start_freq)
    reference_signals = {
        label: _generate_reference_signals(freq, n_samples, SFREQ, n_harmonics)
        for label, freq in STIM_FREQS.items()
    }
    
    cca = CCA(n_components=1)
    class_scores = {}
    for label, ref_sig in reference_signals.items():
        correlations = []
        for sos_filter in filter_bank:
            eeg_filtered = sosfilt(sos_filter, eeg_data, axis=1)
            cca.fit(eeg_filtered.T, ref_sig)
            u, v = cca.transform(eeg_filtered.T, ref_sig)
            corr = np.corrcoef(u.T, v.T)[0, 1]
            correlations.append(corr)
        class_scores[label] = correlations
        
    return class_scores

# --- Main Feature Generation Function ---

def generate_features(base_path, index_csv, task):
    """
    Main function to process a dataset (train, validation, or test) and
    extract features for a given task.
    """
    print(f"--- Generating features for {task} task from {index_csv} ---")
    
    # Select appropriate channels and feature extractor
    if task == "MI":
        channels = MI_CHANNELS
        feature_extractor = extract_mi_features
        label_map = {"Left": 0, "Right": 1}
    elif task == "SSVEP":
        channels = SSVEP_CHANNELS_ALL
        feature_extractor = extract_ssvep_features
        label_encoder = LabelEncoder().fit(LABELS_SSVEP)
    else:
        raise ValueError("Task must be 'MI' or 'SSVEP'")

    # Preprocessing transform
    transform = lambda raw: raw.notch_filter(freqs=50.0, picks="eeg", fir_design="firwin", verbose=False)
    
    dataset = EEG_Dataset(index_csv=index_csv, base_path=base_path, ch_names=channels, task=task, transform=transform)
    
    features = []
    labels = []
    ids = []

    for i in tqdm(range(len(dataset))):
        if "test" in index_csv:
            eeg_tensor = dataset[i]
        else:
            eeg_tensor, label = dataset[i]

        if eeg_tensor is None:
            continue
            
        feature_vec = feature_extractor(eeg_tensor)
        features.append(feature_vec)
        ids.append(dataset.df.iloc[i]['id'])
        
        if "test" not in index_csv:
            if task == "MI":
                labels.append(label_map[label])
            else: # SSVEP
                labels.append(label_encoder.transform([label])[0])

    if "test" in index_csv:
        return np.array(features), np.array(ids)
    else:
        return np.array(features), np.array(labels)

if __name__ == '__main__':
    # This part is for demonstration and allows running the script directly
    # to generate and save feature files.
    DATA_PATH = './' # Assume data is in the current directory
    
    # Generate and save MI features
    X_train_mi, y_train_mi = generate_features(DATA_PATH, 'train_metadata.csv', 'MI')
    X_val_mi, y_val_mi = generate_features(DATA_PATH, 'validation_metadata.csv', 'MI')
    np.savez('features_mi.npz', X_train=X_train_mi, y_train=y_train_mi, X_val=X_val_mi, y_val=y_val_mi)
    print("Saved MI features to features_mi.npz")
    
    # Generate and save SSVEP features
    X_train_ssvep, y_train_ssvep = generate_features(DATA_PATH, 'train_metadata.csv', 'SSVEP')
    X_val_ssvep, y_val_ssvep = generate_features(DATA_PATH, 'validation_metadata.csv', 'SSVEP')
    np.savez('features_ssvep.npz', X_train=X_train_ssvep, y_train=y_train_ssvep, X_val=X_val_ssvep, y_val=y_val_ssvep)
    print("Saved SSVEP features to features_ssvep.npz")
