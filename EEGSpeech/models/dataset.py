import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

class EEGDataset(Dataset):
    """PyTorch Dataset for EEG data"""
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def create_synthetic_eeg_data(n_samples=500, n_channels=14, time_points=1000):
    """
    Create synthetic EEG data with more realistic noise and variability.
    """
    # Vowel phonemes as classes
    classes = ['a', 'e', 'i', 'o', 'u']
    n_classes = len(classes)
    
    # Increase background noise
    X = np.random.randn(n_samples, n_channels, time_points) * 0.3  # More noise
    
    for i in range(n_samples):
        class_idx = i % n_classes
        
        # Add variation to formant frequencies (±10%)
        f1_base = [800, 600, 300, 500, 300][class_idx]
        f2_base = [1200, 1800, 2300, 900, 800][class_idx]
        
        f1 = f1_base * (0.9 + 0.2 * np.random.rand())
        f2 = f2_base * (0.9 + 0.2 * np.random.rand())
        
        t = np.arange(time_points) / 1000
        
        # Add random phase shifts
        phase_shift = np.random.rand() * np.pi
        pattern1 = 0.5 * np.sin(2 * np.pi * f1 * t + phase_shift)
        pattern2 = 0.3 * np.sin(2 * np.pi * f2 * t + phase_shift)
        combined = pattern1 + pattern2
        
        # Add to channels with more variability
        frontal_channels = [0, 1, 2, 3]
        temporal_channels = [7, 8, 9, 10]
        
        for c in frontal_channels:
            # Different amplitude and phase per channel
            channel_phase = np.random.rand() * np.pi/4
            X[i, c, :] += combined * (0.5 + 0.3 * np.random.rand()) * np.sin(t + channel_phase)
        
        for c in temporal_channels:
            channel_phase = np.random.rand() * np.pi/4
            X[i, c, :] += combined * (0.7 + 0.3 * np.random.rand()) * np.sin(t + channel_phase)
    
    # Create labels
    y = np.array([i % n_classes for i in range(n_samples)])
    
    return X, y, classes

def prepare_data_loaders(X, y, batch_size=32):
    """Prepare train, validation, and test data loaders"""
    # Split data into train, validation, and test sets
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    
    # Create datasets
    train_dataset = EEGDataset(X_train, y_train)
    val_dataset = EEGDataset(X_val, y_val)
    test_dataset = EEGDataset(X_test, y_test)
    
    # Create data loaders
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader
