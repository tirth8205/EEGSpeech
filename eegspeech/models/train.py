import torch
import torch.nn as nn
import torch.optim as optim
import time
import numpy as np

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=50, patience=5):
    """Trains model with early stopping"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    best_loss = float('inf')
    patience_counter = 0
    history = {'train_loss': [], 'val_loss': [], 'val_acc': []}
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        epoch_loss = 0.0
        start_time = time.time()
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item() * inputs.size(0)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                val_loss += criterion(outputs, labels).item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        # Record metrics
        train_loss = epoch_loss / len(train_loader.dataset)
        val_loss = val_loss / len(val_loader.dataset)
        val_acc = correct / total
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # Early stopping check
        if val_loss < best_loss:
            best_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            patience_counter += 1
            
        print(f'Epoch {epoch+1}/{num_epochs} | '
              f'Train Loss: {train_loss:.4f} | '
              f'Val Loss: {val_loss:.4f} | '
              f'Val Acc: {val_acc:.4f} | '
              f'Time: {time.time()-start_time:.1f}s')
        
        if patience_counter >= patience:
            print(f'Early stopping at epoch {epoch+1}')
            break
    
    model.load_state_dict(torch.load('best_model.pth'))
    return history

def evaluate_model(model, test_loader):
    """Evaluates model on test set"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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

def train_cli(epochs=50, batch_size=32, lr=0.001, output_path='models/eeg_speech_classifier.pth'):
    """Command-line training interface"""
    from .model import EEGSpeechClassifier
    from .dataset import create_synthetic_eeg_data, prepare_data_loaders
    
    # Set seeds
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Generate data
    X, y, _ = create_synthetic_eeg_data()
    train_loader, val_loader, test_loader = prepare_data_loaders(X, y, batch_size)
    
    # Initialize model
    model = EEGSpeechClassifier(X.shape[1], 5)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0.001)
    
    # Train
    history = train_model(model, train_loader, val_loader, criterion, optimizer, epochs)
    
    # Evaluate
    evaluate_model(model, test_loader)
    
    # Save model
    torch.save(model.state_dict(), output_path)
    print(f'Model saved to {output_path}')
    
    return model, history
