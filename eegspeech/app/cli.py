import argparse
from eegspeech.models.train import train_cli, evaluate_model
from eegspeech.models.dataset import load_eeg_data, prepare_data_loaders
from eegspeech.models.model import EEGSpeechClassifier
from eegspeech.models.utils import calculate_model_size, plot_grad_cam_summary

def main():
    """Command-line interface entry point for EEGSpeech"""
    parser = argparse.ArgumentParser(description="EEGSpeech: Neural Speech Decoding from EEG Signals")
    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # Train command
    train_parser = subparsers.add_parser('train', help='Train the model')
    train_parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    train_parser.add_argument('--batch-size', type=int, default=32, help='Batch size for training')
    train_parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    train_parser.add_argument('--output', type=str, default='eeg_speech_classifier.pth', 
                             help='Output model path')
    train_parser.add_argument('--data-type', choices=['synthetic', 'real'], default='synthetic',
                             help='Type of EEG data to use')
    train_parser.add_argument('--file-path', type=str, help='Path to real EEG data file (EDF)')
    train_parser.add_argument('--kfold', action='store_true', help='Use k-fold cross-validation')

    # Evaluate command
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate a trained model')
    eval_parser.add_argument('--model-path', type=str, default='eeg_speech_classifier.pth',
                            help='Path to trained model')
    eval_parser.add_argument('--data-type', choices=['synthetic', 'real'], default='synthetic',
                            help='Type of EEG data to use')
    eval_parser.add_argument('--file-path', type=str, help='Path to real EEG data file (EDF)')
    eval_parser.add_argument('--batch-size', type=int, default=32, help='Batch size for evaluation')

    # Predict command
    predict_parser = subparsers.add_parser('predict', help='Make predictions on new EEG data')
    predict_parser.add_argument('--model-path', type=str, default='eeg_speech_classifier.pth',
                               help='Path to trained model')
    predict_parser.add_argument('--input-file', type=str, required=True,
                               help='Path to input EEG data file (EDF)')
    predict_parser.add_argument('--output-file', type=str, default='predictions.txt',
                               help='Path to save prediction results')

    # Analyze command
    analyze_parser = subparsers.add_parser('analyze', help='Analyze model and data')
    analyze_parser.add_argument('--model-path', type=str, default='eeg_speech_classifier.pth',
                               help='Path to trained model')
    analyze_parser.add_argument('--data-type', choices=['synthetic', 'real'], default='synthetic',
                               help='Type of EEG data to use')
    analyze_parser.add_argument('--file-path', type=str, help='Path to real EEG data file (EDF)')
    analyze_parser.add_argument('--batch-size', type=int, default=32, help='Batch size for analysis')
    analyze_parser.add_argument('--grad-cam', action='store_true', help='Generate Grad-CAM visualizations')

    args = parser.parse_args()

    if args.command == 'train':
        train_cli(
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            output_path=args.output,
            data_type=args.data_type,
            file_path=args.file_path,
            kfold=args.kfold
        )

    elif args.command == 'evaluate':
        # Load model
        try:
            model = EEGSpeechClassifier(14, 8)
            model.load_state_dict(torch.load(args.model_path, map_location=torch.device('cpu')))
            model.eval()
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            return
        
        # Load data
        X, y, classes = load_eeg_data(args.data_type, args.file_path, n_samples=1000)
        if X is None:
            return
        _, _, test_loader = prepare_data_loaders(X, y, args.batch_size, augment=False)
        
        # Evaluate
        metrics = evaluate_model(model, test_loader, classes)
        print(f"Evaluation Metrics: {metrics}")

    elif args.command == 'predict':
        # Load model
        try:
            model = EEGSpeechClassifier(14, 8)
            model.load_state_dict(torch.load(args.model_path, map_location=torch.device('cpu')))
            model.eval()
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            return
        
        # Load and preprocess EEG data
        X, _, classes = load_eeg_data('real', args.input_file, n_samples=1000)
        if X is None:
            return
        
        # Make predictions
        predictions = []
        with torch.no_grad():
            for i in range(X.shape[0]):
                input_tensor = torch.FloatTensor(X[i:i+1]).to(torch.device('cpu'))
                output = model(input_tensor)
                probs = torch.nn.functional.softmax(output, dim=1)[0]
                pred = torch.argmax(output, dim=1).item()
                predictions.append((classes[pred], probs[pred].item()))
        
        # Save predictions
        with open(args.output_file, 'w') as f:
            for pred, conf in predictions:
                f.write(f"Predicted: {pred}, Confidence: {conf:.4f}\n")
        print(f"Predictions saved to {args.output_file}")

    elif args.command == 'analyze':
        # Load model
        try:
            model = EEGSpeechClassifier(14, 8)
            model.load_state_dict(torch.load(args.model_path, map_location=torch.device('cpu')))
            model.eval()
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            return
        
        # Load data
        X, y, classes = load_eeg_data(args.data_type, args.file_path, n_samples=1000)
        if X is None:
            return
        _, _, test_loader = prepare_data_loaders(X, y, args.batch_size, augment=False)
        
        # Calculate model size
        params, flops = calculate_model_size(model)
        print(f"Model Complexity: {params/1e6:.2f}M parameters, {flops/1e9:.2f}G FLOPs")
        
        # Generate Grad-CAM if requested
        if args.grad_cam:
            plot_grad_cam_summary(model, test_loader, classes)
            print("Grad-CAM visualizations saved as grad_cam_summary.png/svg")

    else:
        parser.print_help()

if __name__ == "__main__":
    main()