import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
import mne
import scipy.signal as signal
import os

class EEGDataset(Dataset):
    """PyTorch Dataset for EEG speech data"""
    def __init__(self, X, y, augment=False):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)
        self.augment = augment
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        x = self.X[idx]
        if self.augment:
            x = self._augment_data(x)
        return x, self.y[idx]
    
    def _augment_data(self, x):
        """Apply data augmentation: noise, scaling, time warping"""
        # Add Gaussian noise
        noise = np.random.normal(0, 0.1, x.shape)
        x = x + noise
        
        # Random scaling
        scale = np.random.uniform(0.8, 1.2)
        x = x * scale
        
        # Time warping (stretch/compress)
        if np.random.rand() > 0.5:
            x = signal.resample(x, int(x.shape[-1] * np.random.uniform(0.9, 1.1)), axis=-1)
            if x.shape[-1] != self.X.shape[-1]:
                x = np.pad(x, ((0, 0), (0, self.X.shape[-1] - x.shape[-1])), mode='constant')
        
        return x.astype(np.float32)

def create_synthetic_eeg_data(n_samples=1000, n_channels=14, time_points=1000, sfreq=1000):
    """Generates synthetic EEG data with phoneme-specific patterns"""
    classes = ['a', 'e', 'i', 'o', 'u', 'p', 't', 'k']
    X = np.random.randn(n_samples, n_channels, time_points) * 0.3
    y = np.array([i % len(classes) for i in range(n_samples)])
    
    t = np.arange(time_points) / sfreq
    for i in range(n_samples):
        class_idx = y[i]
        # Phoneme-specific formant frequencies and patterns
        patterns = [
            # Vowels: F1, F2 (formant frequencies in Hz)
            {'f1': 800, 'f2': 1200, 'amp': 0.7},  # /a/
            {'f1': 600, 'f2': 1800, 'amp': 0.6},  # /e/
            {'f1': 300, 'f2': 2300, 'amp': 0.5},  # /i/
            {'f1': 500, 'f2': 900, 'amp': 0.7},   # /o/
            {'f1': 300, 'f2': 800, 'amp': 0.6},   # /u/
            # Consonants: burst frequency, duration
            {'burst': 2000, 'dur': 0.05, 'amp': 0.9},  # /p/
            {'burst': 1500, 'dur': 0.03, 'amp': 0.8},  # /t/
            {'burst': 1000, 'dur': 0.04, 'amp': 0.85}  # /k/
        ]
        
        pattern = patterns[class_idx]
        if class_idx < 5:  # Vowels
            f1 = pattern['f1'] * np.random.uniform(0.9, 1.1)
            f2 = pattern['f2'] * np.random.uniform(0.9, 1.1)
            signal = pattern['amp'] * (0.5 * np.sin(2 * np.pi * f1 * t) + 
                                    0.3 * np.sin(2 * np.pi * f2 * t + np.pi/4))
        else:  # Consonants
            burst = pattern['burst'] * np.random.uniform(0.9, 1.1)
            dur = pattern['dur']
            signal = np.zeros(time_points)
            burst_start = int(time_points * 0.2)
            burst_end = int(burst_start + dur * sfreq)
            signal[burst_start:burst_end] = pattern['amp'] * np.sin(2 * np.pi * burst * t[:burst_end-burst_start])
        
        # Apply to specific channels
        X[i, [0, 1, 2, 3], :] += signal * 0.7  # Frontal channels
        X[i, [7, 8, 9, 10], :] += signal * 0.9  # Temporal channels
        X[i, [4, 5, 6], :] += signal * 0.5  # Parietal channels
    
    return X, y, classes

def preprocess_real_eeg(file_path, sfreq=1000, n_channels=14, time_points=1000):
    """Preprocess real EEG data from EDF or similar files"""
    try:
        # Load EEG data using MNE
        raw = mne.io.read_raw_edf(file_path, preload=True, verbose=False)
        
        # Band-pass filter (1-50 Hz)
        raw.filter(l_freq=1, h_freq=50, method='fir')
        
        # Select first n_channels (or map to standard 10-20 system if needed)
        channels = raw.ch_names[:min(n_channels, len(raw.ch_names))]
        raw.pick_channels(channels)
        
        # Resample to target sampling frequency
        raw.resample(sfreq)
        
        # Extract epochs (assuming 1-second segments)
        data = raw.get_data()
        n_samples = data.shape[1] // time_points
        X = np.zeros((n_samples, n_channels, time_points))
        
        for i in range(n_samples):
            start = i * time_points
            end = start + time_points
            X[i, :len(channels), :] = data[:len(channels), start:end]
        
        # Placeholder labels (to be replaced with real annotations)
        y = np.zeros(n_samples, dtype=int)  # Dummy labels for demo
        classes = ['unknown']
        
        return X, y, classes
    except Exception as e:
        print(f"Error processing EEG file: {str(e)}")
        return None, None, None

def prepare_data_loaders(X, y, batch_size=32, augment=True):
    """Creates train/val/test loaders with augmentation"""
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    
    return (
        torch.utils.data.DataLoader(
            EEGDataset(X_train, y_train, augment=augment),
            batch_size=batch_size,
            shuffle=True,
            num_workers=2
        ),
        torch.utils.data.DataLoader(
            EEGDataset(X_val, y_val, augment=False),
            batch_size=batch_size,
            num_workers=2
        ),
        torch.utils.data.DataLoader(
            EEGDataset(X_test, y_test, augment=False),
            batch_size=batch_size,
            num_workers=2
        )
    )

def load_eeg_data(data_type='synthetic', file_path=None, n_samples=1000, n_channels=14, time_points=1000, sfreq=1000):
    """Unified interface for loading synthetic or real EEG data"""
    if data_type == 'synthetic':
        return create_synthetic_eeg_data(n_samples, n_channels, time_points, sfreq)
    elif data_type == 'real' and file_path and os.path.exists(file_path):
        return preprocess_real_eeg(file_path, sfreq, n_channels, time_points)
    else:
        raise ValueError("Invalid data_type or file_path not provided for real EEG data")