import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
from sklearn.model_selection import KFold
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=50, patience=5, device=None):
    """Trains model with early stopping"""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    best_loss = float('inf')
    patience_counter = 0
    history = {'train_loss': [], 'val_loss': [], 'val_acc': [], 'val_f1': []}
    
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
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                val_loss += criterion(outputs, labels).item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Calculate metrics
        train_loss = epoch_loss / len(train_loader.dataset)
        val_loss = val_loss / len(val_loader.dataset)
        val_acc = correct / total
        precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted', zero_division=0)
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_f1'].append(f1)
        
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
              f'Val F1: {f1:.4f} | '
              f'Time: {time.time()-start_time:.1f}s')
        
        if patience_counter >= patience:
            print(f'Early stopping at epoch {epoch+1}')
            break
    
    model.load_state_dict(torch.load('best_model.pth'))
    return history

def evaluate_model(model, test_loader, classes, device=None):
    """Evaluates model on test set with detailed metrics"""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate metrics
    accuracy = correct / total
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted', zero_division=0)
    
    # Plot confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig('confusion_matrix.png')
    plt.close()
    
    print(f'Test Accuracy: {accuracy:.4f}')
    print(f'Test Precision: {precision:.4f}')
    print(f'Test Recall: {recall:.4f}')
    print(f'Test F1-Score: {f1:.4f}')
    
    return {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1}

def train_kfold(model_class, X, y, classes, n_splits=5, epochs=50, batch_size=32, lr=0.001):
    """Performs k-fold cross-validation"""
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_results = []
    
    for fold, (train_idx, val_idx) in enumerate(kfold.split(X)):
        print(f'\nFold {fold+1}/{n_splits}')
        
        # Split data
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # Create data loaders
        from .dataset import prepare_data_loaders
        train_loader, val_loader, _ = prepare_data_loaders(X_train, y_train, batch_size, augment=True)
        
        # Initialize model
        model = model_class(X.shape[1], len(classes))
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0.001)
        
        # Train
        history = train_model(model, train_loader, val_loader, criterion, optimizer, epochs)
        
        # Evaluate
        _, _, test_loader = prepare_data_loaders(X_train, y_train, batch_size, augment=False)
        metrics = evaluate_model(model, test_loader, classes)
        fold_results.append(metrics)
    
    # Average metrics
    avg_metrics = {
        key: np.mean([r[key] for r in fold_results])
        for key in ['accuracy', 'precision', 'recall', 'f1']
    }
    print(f'\nK-Fold Average Metrics:')
    for key, value in avg_metrics.items():
        print(f'{key.capitalize()}: {value:.4f}')
    
    return avg_metrics, fold_results

def train_cli(epochs=50, batch_size=32, lr=0.001, output_path='models/eeg_speech_classifier.pth', data_type='synthetic', file_path=None, kfold=False):
    """Command-line training interface with k-fold and hyperparameter tuning"""
    from .model import EEGSpeechClassifier
    from .dataset import load_eeg_data, prepare_data_loaders
    
    # Set seeds
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Load data
    X, y, classes = load_eeg_data(data_type, file_path)
    if X is None:
        return None, None
    
    # Hyperparameter grid
    lrs = [lr] if kfold else [0.0005, 0.001, 0.002]
    batch_sizes = [batch_size] if kfold else [16, 32, 64]
    
    best_metrics = None
    best_model = None
    best_params = None
    
    for lr in lrs:
        for bs in batch_sizes:
            print(f'\nTraining with lr={lr}, batch_size={bs}')
            
            if kfold:
                # Perform k-fold cross-validation
                metrics, _ = train_kfold(EEGSpeechClassifier, X, y, classes, n_splits=5, epochs=epochs, batch_size=bs, lr=lr)
            else:
                # Single train-val-test split
                train_loader, val_loader, test_loader = prepare_data_loaders(X, y, bs, augment=True)
                model = EEGSpeechClassifier(X.shape[1], len(classes))
                criterion = nn.CrossEntropyLoss()
                optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0.001)
                history = train_model(model, train_loader, val_loader, criterion, optimizer, epochs)
                metrics = evaluate_model(model, test_loader, classes)
            
            # Update best model
            if best_metrics is None or metrics['f1'] > best_metrics['f1']:
                best_metrics = metrics
                best_model = model
                best_params = {'lr': lr, 'batch_size': bs}
    
    # Save best model
    if best_model:
        torch.save(best_model.state_dict(), output_path)
        print(f'Model saved to {output_path}')
        print(f'Best parameters: lr={best_params["lr"]}, batch_size={best_params["batch_size"]}')
    
    return best_model, best_metrics