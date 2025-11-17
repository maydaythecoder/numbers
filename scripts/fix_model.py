"""Fix model compatibility issues by rebuilding with the current architecture."""
import os
import sys
from pathlib import Path

import tensorflow as tf

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from digit_recognition import MODEL_PATH, build_model

def fix_model():
    old_model_path = MODEL_PATH
    backup_path = MODEL_PATH.parent / 'handwritten_digits.model.keras.backup'
    
    if not os.path.exists(old_model_path):
        print(f"Model file {old_model_path} not found!")
        return
    
    print("Backing up old model...")
    if os.path.exists(backup_path):
        os.remove(backup_path)
    os.rename(old_model_path, backup_path)
    
    print("Rebuilding model with correct architecture...")
    model = build_model()
    
    print("Loading weights from old model...")
    try:
        # Try to load just the weights
        old_model = tf.keras.models.load_model(backup_path, compile=False)
        # Copy weights layer by layer
        for i, layer in enumerate(model.layers[1:], 1):  # Skip Flatten layer
            if i < len(old_model.layers):
                try:
                    layer.set_weights(old_model.layers[i].get_weights())
                    print(f"  Loaded weights for layer {i}: {layer.name}")
                except Exception as e:
                    print(f"  Warning: Could not load weights for layer {i}: {e}")
    except Exception as e:
        print(f"Could not load weights from old model: {e}")
        print("You'll need to retrain the model. Run: python scripts/recognition.py")
        # Restore backup
        os.rename(backup_path, old_model_path)
        return
    
    print("Compiling model...")
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    print("Saving fixed model...")
    model.save(old_model_path)
    print(f"✓ Model fixed and saved! Old model backed up as {backup_path}")
    print("You can now run the web app or GUI app.")

if __name__ == '__main__':
    fix_model()
