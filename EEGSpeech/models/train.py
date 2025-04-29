import torch
import torch.nn as nn
import torch.optim as optim
import time
import os
import numpy as np

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10, patience=3):
    """Train the model with early stopping"""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model.to(device)
    
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_acc': []
    }
    
    # Early stopping variables
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    
    start_time = time.time()
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
        
        epoch_loss = running_loss / len(train_loader.dataset)
        history['train_loss'].append(epoch_loss)
        
        # Validation phase
        model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                running_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        epoch_loss = running_loss / len(val_loader.dataset)
        epoch_acc = correct / total
        
        history['val_loss'].append(epoch_loss)
        history['val_acc'].append(epoch_acc)
        
        elapsed = time.time() - start_time
        print(f'Epoch {epoch+1}/{num_epochs} [{elapsed:.1f}s], '
              f'Train Loss: {history["train_loss"][-1]:.4f}, '
              f'Val Loss: {history["val_loss"][-1]:.4f}, '
              f'Val Acc: {history["val_acc"][-1]:.4f}')
        
        # Early stopping check
        if epoch_loss < best_val_loss:
            best_val_loss = epoch_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
            print("New best model saved")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                model.load_state_dict(best_model_state)
                break
    
    return history

def evaluate_model(model, test_loader):
    """Evaluate the model on test data"""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = correct / total
    print(f'Test Accuracy: {accuracy:.4f}')
    return accuracy

def train_cli(epochs=10, batch_size=32, learning_rate=0.001, output_path="models/eeg_speech_classifier.pth"):
    """CLI training function"""
    from .model import EEGSpeechClassifier
    from .dataset import create_synthetic_eeg_data, prepare_data_loaders
    
    # Set random seeds
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Generate data
    X, y, class_names = create_synthetic_eeg_data()
    train_loader, val_loader, test_loader = prepare_data_loaders(X, y, batch_size)
    
    # Initialize model
    model = EEGSpeechClassifier(X.shape[1], len(class_names))
    
    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.001)
    
    # Train model
    history = train_model(model, train_loader, val_loader, criterion, optimizer, epochs)
    
    # Save model
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(model.state_dict(), output_path)
    print(f"Model saved to {output_path}")
    
    return model, history
