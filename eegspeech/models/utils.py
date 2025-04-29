import matplotlib.pyplot as plt
import torch
import numpy as np

def plot_training_history(history):
    """Plots training and validation metrics"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Loss plot
    ax1.plot(history['train_loss'], label='Train')
    ax1.plot(history['val_loss'], label='Validation')
    ax1.set_title('Loss Curve')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend()
    
    # Accuracy plot
    ax2.plot(history['val_acc'], label='Validation')
    ax2.set_title('Accuracy')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.close()

def visualize_eeg_and_predictions(model, test_loader, class_names, num_samples=3):
    """Visualizes EEG samples and model predictions"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Get batch
    inputs, labels = next(iter(test_loader))
    inputs = inputs[:num_samples].to(device)
    
    # Predictions
    with torch.no_grad():
        outputs = model(inputs)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        _, preds = torch.max(outputs, 1)
    
    # Plot
    fig, axes = plt.subplots(num_samples, 1, figsize=(10, 3*num_samples))
    for i in range(num_samples):
        if num_samples == 1:
            ax = axes
        else:
            ax = axes[i]
            
        # Plot first 4 channels
        for ch in range(4):
            ax.plot(inputs[i][ch].cpu().numpy(), label=f'Ch {ch}')
        
        # Add title with results
        correct = preds[i] == labels[i]
        title_color = 'green' if correct else 'red'
        ax.set_title(f'True: {class_names[labels[i]]} | Pred: {class_names[preds[i]]} | Confidence: {probs[i][preds[i]]:.2f}',
                    color=title_color)
        ax.legend()
    
    plt.tight_layout()
    plt.savefig('eeg_predictions.png')
    plt.close()

def calculate_model_size(model):
    """Prints number of trainable parameters"""
    total = sum(p.numel() for p in model.parameters())
    print(f"Model Parameters: {total/1e3:.1f}K")
    return total
