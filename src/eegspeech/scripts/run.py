import sys
import subprocess
import os

def check_model_exists():
    return os.path.exists('eeg_speech_classifier.pth')

def main():
    print("EEG Speech Decoder - Control Panel")
    print("----------------------------------")
    
    if check_model_exists():
        print("✅ Trained model found!")
    else:
        print("⚠️ No trained model found. You should train the model first.")
    
    print("\nOptions:")
    print("1. Train the model")
    print("2. Launch the interactive UI")
    print("3. Exit")
    
    choice = input("\nEnter your choice (1-3): ")
    
    if choice == '1':
        print("\nTraining model...")
        subprocess.run([sys.executable, 'main.py'])
        print("\nTraining complete!")
        input("Press Enter to continue...")
        main()
    elif choice == '2':
        print("\nLaunching interactive UI...")
        subprocess.run([sys.executable, '-m', 'streamlit', 'run', 'app.py'])
    elif choice == '3':
        print("\nExiting...")
        sys.exit(0)
    else:
        print("\nInvalid choice! Please try again.")
        input("Press Enter to continue...")
        main()

if __name__ == "__main__":
    main()
