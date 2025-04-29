import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

class EEGDataset(Dataset):
    """PyTorch Dataset for EEG speech data"""
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def create_synthetic_eeg_data(n_samples=500, n_channels=14, time_points=1000):
    """Generates synthetic EEG data with phoneme-specific patterns"""
    classes = ['a', 'e', 'i', 'o', 'u']
    X = np.random.randn(n_samples, n_channels, time_points) * 0.3
    
    for i in range(n_samples):
        class_idx = i % 5
        # Create frequency-based patterns
        f1 = [800, 600, 300, 500, 300][class_idx] * (0.9 + 0.2*np.random.rand())
        f2 = [1200, 1800, 2300, 900, 800][class_idx] * (0.9 + 0.2*np.random.rand())
        
        t = np.arange(time_points)/1000
        signal = 0.5*np.sin(2*np.pi*f1*t) + 0.3*np.sin(2*np.pi*f2*t + np.pi/4)
        
        # Add to relevant channels
        X[i, [0,1,2,3], :] += signal * 0.7  # Frontal channels
        X[i, [7,8,9,10], :] += signal * 0.9  # Temporal channels
    
    y = np.array([i%5 for i in range(n_samples)])
    return X, y, classes

def prepare_data_loaders(X, y, batch_size=32):
    """Creates train/val/test loaders"""
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5)
    
    return (
        torch.utils.data.DataLoader(EEGDataset(X_train, y_train), batch_size, shuffle=True),
        torch.utils.data.DataLoader(EEGDataset(X_val, y_val), batch_size),
        torch.utils.data.DataLoader(EEGDataset(X_test, y_test), batch_size)
    )
