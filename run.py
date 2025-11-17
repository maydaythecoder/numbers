#!/usr/bin/env python3
"""
Unified launcher script for Handwritten Digit Recognition applications.
Run this script to launch any of the available interfaces.
"""
import os
import sys
import subprocess
import argparse

def check_venv():
    """Check if virtual environment is activated, activate if not."""
    if not hasattr(sys, 'real_prefix') and not (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        venv_path = os.path.join(os.path.dirname(__file__), 'venv', 'bin', 'activate')
        if os.path.exists(venv_path):
            print("⚠️  Virtual environment not activated. Please run:")
            print(f"   source {venv_path}")
            print("   Or use: python run.py [option]")
            return False
    return True

def check_model():
    """Check if model file exists."""
    model_path = 'handwritten_digits.model.keras'
    if not os.path.exists(model_path):
        print("❌ Model file not found!")
        print("   Training new model...")
        try:
            subprocess.run([sys.executable, 'recognition.py'], check=True)
            print("✅ Model trained successfully!")
        except subprocess.CalledProcessError:
            print("❌ Failed to train model. Please run: python recognition.py")
            return False
    return True

def run_web():
    """Run the web interface."""
    print("🌐 Starting web interface...")
    print("   Open http://localhost:5000 in your browser")
    print("   Press Ctrl+C to stop\n")
    try:
        subprocess.run([sys.executable, 'web_app.py'])
    except KeyboardInterrupt:
        print("\n\n👋 Web interface stopped.")

def run_gui():
    """Run the desktop GUI."""
    print("🖥️  Starting desktop GUI...")
    print("   Close the window to exit\n")
    try:
        subprocess.run([sys.executable, 'gui_app.py'])
    except KeyboardInterrupt:
        print("\n\n👋 GUI stopped.")

def run_recognition():
    """Run the original recognition script."""
    print("📸 Processing images from digits/ folder...\n")
    try:
        subprocess.run([sys.executable, 'recognition.py'])
    except KeyboardInterrupt:
        print("\n\n👋 Recognition stopped.")

def main():
    parser = argparse.ArgumentParser(
        description='Handwritten Digit Recognition - Unified Launcher',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run.py web      # Start web interface
  python run.py gui      # Start desktop GUI
  python run.py images   # Process images from digits/ folder
  python run.py          # Show interactive menu
        """
    )
    
    parser.add_argument(
        'mode',
        nargs='?',
        choices=['web', 'gui', 'images'],
        help='Application mode to run'
    )
    
    args = parser.parse_args()
    
    # Check model exists
    if not check_model():
        sys.exit(1)
    
    # If mode specified, run directly
    if args.mode:
        if args.mode == 'web':
            run_web()
        elif args.mode == 'gui':
            run_gui()
        elif args.mode == 'images':
            run_recognition()
        return
    
    # Interactive menu
    print("=" * 50)
    print("  Handwritten Digit Recognition - Launcher")
    print("=" * 50)
    print()
    print("Select an option:")
    print("  1. Web Interface (http://localhost:5000)")
    print("  2. Desktop GUI")
    print("  3. Process Images (from digits/ folder)")
    print("  4. Exit")
    print()
    
    while True:
        try:
            choice = input("Enter choice (1-4): ").strip()
            
            if choice == '1':
                run_web()
                break
            elif choice == '2':
                run_gui()
                break
            elif choice == '3':
                run_recognition()
                break
            elif choice == '4':
                print("👋 Goodbye!")
                break
            else:
                print("❌ Invalid choice. Please enter 1, 2, 3, or 4.")
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except EOFError:
            print("\n\n👋 Goodbye!")
            break

if __name__ == '__main__':
    main()

