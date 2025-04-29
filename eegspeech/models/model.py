import torch
import torch.nn as nn

class EEGSpeechClassifier(nn.Module):
    """Temporal CNN for EEG speech decoding"""
    def __init__(self, n_channels=14, n_classes=5, time_points=1000):
        super().__init__()
        
        # Feature extraction
        self.features = nn.Sequential(
            nn.Conv1d(n_channels, 16, 5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(4),
            nn.Conv1d(16, 32, 5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(4),
            nn.Dropout(0.3)
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(32 * (time_points//16), 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, n_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)
