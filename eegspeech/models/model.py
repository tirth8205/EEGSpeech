import torch
import torch.nn as nn

class EEGSpeechClassifier(nn.Module):
    """Hybrid CNN-LSTM for EEG speech decoding with batch normalization"""
    def __init__(self, n_channels=14, n_classes=8, time_points=1000):
        super().__init__()
        
        # Feature extraction with CNN
        self.features = nn.Sequential(
            nn.Conv1d(n_channels, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(4),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(4),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.4)
        )
        
        # LSTM for temporal dependencies
        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=256,
            num_layers=2,
            batch_first=True,
            dropout=0.3,
            bidirectional=True
        )
        
        # Calculate LSTM output size
        self.conv_output_size = time_points // 32  # After three pooling layers
        self.lstm_output_size = 256 * 2  # Bidirectional LSTM
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(self.lstm_output_size * self.conv_output_size, 512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, n_classes)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def forward(self, x):
        # CNN feature extraction
        x = self.features(x)
        
        # Reshape for LSTM: (batch, channels, seq_len) -> (batch, seq_len, channels)
        x = x.permute(0, 2, 1)
        
        # LSTM processing
        x, _ = self.lstm(x)
        
        # Flatten for classifier
        x = x.contiguous().view(x.size(0), -1)
        
        # Classification
        return self.classifier(x)
    
    def _initialize_weights(self):
        """Initialize model weights for better training"""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def get_model_size(self):
        """Returns number of trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)