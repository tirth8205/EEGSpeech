import torch
import torch.nn as nn

class EEGSpeechClassifier(nn.Module):
    """Lightweight neural network for classifying speech phonemes from EEG data."""
    def __init__(self, n_channels, n_classes, time_points=1000):
        super(EEGSpeechClassifier, self).__init__()
        
        # Calculate feature dimension after two pooling layers
        pooled_length = time_points // 16  # Two MaxPool1d with stride=4
        self.feature_dim = 16 * pooled_length
        
        # Smaller feature extractor
        self.feature_extractor = nn.Sequential(
            nn.Conv1d(n_channels, 8, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4, stride=4),
            nn.Conv1d(8, 16, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4, stride=4),
            nn.Dropout(0.3)  # Increased dropout
        )
        
        # Smaller classifier
        self.classifier = nn.Sequential(
            nn.Linear(self.feature_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, n_classes)
        )
    
    def forward(self, x):
        # x shape: [batch_size, channels, time]
        features = self.feature_extractor(x)
        features = features.view(features.size(0), -1)
        output = self.classifier(features)
        return output