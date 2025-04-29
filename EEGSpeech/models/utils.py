import matplotlib.pyplot as plt
import torch
import numpy as np

def plot_training_history(history):
    """Plot training and validation metrics"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot loss
    ax1.plot(history['train_loss'], label='Train Loss')
    ax1.plot(history['val_loss'], label='Val Loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.set_title('Training and Validation Loss')
    
    # Plot accuracy
    ax2.plot(history['val_acc'], label='Val Accuracy')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.set_title('Validation Accuracy')
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.close(fig)
    print("Training history plot saved to 'training_history.png'")

def visualize_eeg_and_predictions(model, test_loader, class_names, num_samples=3):
    """Visualize EEG data and model predictions"""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    # Get a batch of data
    dataiter = iter(test_loader)
    inputs, labels = next(dataiter)
    inputs = inputs[:num_samples].to(device)
    labels = labels[:num_samples]
    
    # Get predictions
    with torch.no_grad():
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
    
    # Plot
    fig, axes = plt.subplots(num_samples, 1, figsize=(10, 4*num_samples))
    if num_samples == 1:
        axes = [axes]
    
    for i in range(num_samples):
        eeg_data = inputs[i].cpu().numpy()
        
        # Plot EEG channels (just a subset for visibility)
        for j in range(min(4, eeg_data.shape[0])):
            axes[i].plot(eeg_data[j], label=f'Channel {j}')
        
        # Add title with true and predicted class
        correct = "✓" if preds[i].item() == labels[i].item() else "✗"
        axes[i].set_title(f'True: {class_names[labels[i]]}, Predicted: {class_names[preds[i].item()]} {correct}')
        
        axes[i].set_xlabel('Time (samples)')
        axes[i].set_ylabel('Amplitude')
        axes[i].legend()
    
    plt.tight_layout()
    plt.savefig('eeg_predictions.png')
    plt.close(fig)
    print("EEG predictions plot saved to 'eeg_predictions.png'")

def calculate_model_size(model):
    """Calculate the number of parameters in the model"""
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    return total_params
