import argparse
from eegspeech.models.train import train_cli

def main():
    """Command-line interface entry point"""
    parser = argparse.ArgumentParser(description="EEGSpeech CLI")
    subparsers = parser.add_subparsers(dest='command')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train the model')
    train_parser.add_argument('--epochs', type=int, default=50,
                             help='Number of training epochs')
    train_parser.add_argument('--batch-size', type=int, default=32,
                             help='Batch size for training')
    train_parser.add_argument('--lr', type=float, default=0.001,
                             help='Learning rate')
    train_parser.add_argument('--output', type=str, 
                             default='eeg_speech_classifier.pth',
                             help='Output model path')
    
    args = parser.parse_args()
    
    if args.command == 'train':
        train_cli(epochs=args.epochs, 
                 batch_size=args.batch_size,
                 lr=args.lr,
                 output_path=args.output)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
