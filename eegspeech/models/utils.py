import matplotlib.pyplot as plt
import torch
import numpy as np
from torch.nn import functional as F
import torchinfo
import seaborn as sns

def plot_training_history(history):
    """Plots training and validation metrics including F1-score"""
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4))
    
    # Loss plot
    ax1.plot(history['train_loss'], label='Train Loss', color='blue')
    ax1.plot(history['val_loss'], label='Validation Loss', color='orange')
    ax1.set_title('Loss Curve')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Accuracy plot
    ax2.plot(history['val_acc'], label='Validation Accuracy', color='green')
    ax2.set_title('Accuracy Curve')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    # F1-score plot
    ax3.plot(history['val_f1'], label='Validation F1-Score', color='purple')
    ax3.set_title('F1-Score Curve')
    ax3.set_xlabel('Epochs')
    ax3.set_ylabel('F1-Score')
    ax3.legend()
    ax3.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=300)
    plt.savefig('training_history.svg')
    plt.close()

def visualize_eeg_and_predictions(model, test_loader, class_names, num_samples=3):
    """Visualizes EEG samples and model predictions with Grad-CAM"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    # Get batch
    inputs, labels = next(iter(test_loader))
    inputs = inputs[:num_samples].to(device)
    inputs.requires_grad = True
    
    # Predictions
    with torch.no_grad():
        outputs = model(inputs)
        probs = F.softmax(outputs, dim=1)
        _, preds = torch.max(outputs, 1)
    
    # Grad-CAM for the first sample
    cam = compute_grad_cam(model, inputs[0:1], preds[0], device)
    
    # Plot
    fig, axes = plt.subplots(num_samples + 1, 1, figsize=(10, 3 * (num_samples + 1)))
    for i in range(num_samples):
        ax = axes[i] if num_samples > 1 else axes
        # Plot first 4 channels
        for ch in range(4):
            ax.plot(inputs[i][ch].cpu().numpy(), label=f'Ch {ch}', alpha=0.7)
        
        # Add title with results
        correct = preds[i] == labels[i]
        title_color = 'green' if correct else 'red'
        ax.set_title(
            f'True: {class_names[labels[i]]} | Pred: {class_names[preds[i]]} | '
            f'Confidence: {probs[i][preds[i]]:.2f}',
            color=title_color
        )
        ax.legend()
        ax.grid(True)
    
    # Plot Grad-CAM
    ax_cam = axes[num_samples] if num_samples > 1 else axes
    sns.heatmap(cam.cpu().numpy(), cmap='viridis', cbar=True, ax=ax_cam)
    ax_cam.set_title('Grad-CAM: Channel Importance over Time')
    ax_cam.set_xlabel('Time Points')
    ax_cam.set_ylabel('Channels')
    
    plt.tight_layout()
    plt.savefig('eeg_predictions.png', dpi=300)
    plt.savefig('eeg_predictions.svg')
    plt.close()

def compute_grad_cam(model, input_tensor, target_class, device):
    """Computes Grad-CAM for EEG input to highlight important channels and time points"""
    model.eval()
    input_tensor.requires_grad = True
    
    # Forward pass
    output = model(input_tensor)
    model.zero_grad()
    
    # Backward pass for target class
    output[0, target_class].backward()
    
    # Get gradients and activations from the last conv layer
    gradients = input_tensor.grad[0]  # Shape: [C, T]
    with torch.no_grad():
        activations = model.features(input_tensor)[0]  # Shape: [C_out, T_out]
    
    # Pool gradients over time
    weights = torch.mean(gradients, dim=1, keepdim=True)  # Shape: [C, 1]
    
    # Compute Grad-CAM
    cam = torch.zeros_like(activations)
    for i in range(weights.size(0)):
        cam += weights[i] * activations
    
    cam = F.relu(cam)
    cam = cam / (cam.max() + 1e-8)  # Normalize
    cam = F.interpolate(cam.unsqueeze(0).unsqueeze(0), size=input_tensor.shape[1:], 
                       mode='bilinear', align_corners=False).squeeze()
    
    return cam

def calculate_model_size(model):
    """Prints number of trainable parameters and FLOPs"""
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model Parameters: {total_params/1e3:.1f}K")
    
    # Calculate FLOPs using torchinfo
    input_size = (1, 14, 1000)  # [batch, channels, time_points]
    summary = torchinfo.summary(model, input_size=input_size, verbose=0)
    print(f"Model FLOPs: {summary.total_mult_adds/1e6:.1f}M")
    
    return total_params, summary.total_mult_adds

def plot_grad_cam_summary(model, test_loader, class_names):
    """Plots Grad-CAM summary for multiple samples"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    inputs, _ = next(iter(test_loader))
    inputs = inputs[:4].to(device)
    inputs.requires_grad = True
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()
    
    for i in range(4):
        with torch.no_grad():
            output = model(inputs[i:i+1])
            _, pred = torch.max(output, 1)
        
        cam = compute_grad_cam(model, inputs[i:i+1], pred, device)
        sns.heatmap(cam.cpu().numpy(), cmap='viridis', cbar=True, ax=axes[i])
        axes[i].set_title(f'Predicted: {class_names[pred]}')
        axes[i].set_xlabel('Time Points')
        axes[i].set_ylabel('Channels')
    
    plt.tight_layout()
    plt.savefig('grad_cam_summary.png', dpi=300)
    plt.savefig('grad_cam_summary.svg')
    plt.close()