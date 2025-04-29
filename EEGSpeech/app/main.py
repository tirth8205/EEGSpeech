import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Import from project modules (package-relative)
from EEGSpeech.models.model import EEGSpeechClassifier
from EEGSpeech.models.dataset import create_synthetic_eeg_data, prepare_data_loaders
from EEGSpeech.models.train import train_model, evaluate_model
from EEGSpeech.models.utils import plot_training_history, visualize_eeg_and_predictions, calculate_model_size

def main():
    # Set random seed for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Generate synthetic EEG data
    print("Creating synthetic EEG data for speech phoneme classification...")
    X, y, class_names = create_synthetic_eeg_data(n_samples=500, n_channels=14, time_points=1000)
    
    # Prepare data loaders
    train_loader, val_loader, test_loader = prepare_data_loaders(X, y, batch_size=32)
    print(f"Data split: {X.shape[0]} training, {len(val_loader.dataset)} validation, {len(test_loader.dataset)} test samples")
    
    # Create model
    n_channels = X.shape[1]
    n_classes = len(class_names)
    time_points = X.shape[2]
    
    print(f"Creating model for {n_channels} EEG channels and {n_classes} phoneme classes...")
    model = EEGSpeechClassifier(n_channels, n_classes, time_points)
    
    # Print model summary
    print("\nModel Structure:")
    print(model)
    calculate_model_size(model)
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.001)
    
    # Train model
    print("\nTraining model...")
    history = train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10)
    
    # Evaluate
    print("\nEvaluating model on test set...")
    evaluate_model(model, test_loader)
    
    # Visualize results
    print("\nPlotting training history...")
    plot_training_history(history)
    
    print("\nVisualizing EEG predictions...")
    visualize_eeg_and_predictions(model, test_loader, class_names)
    
    # Save model to root directory
    torch.save(model.state_dict(), 'eeg_speech_classifier.pth')
    print("\nModel saved to 'eeg_speech_classifier.pth'")

if __name__ == "__main__":
    main()