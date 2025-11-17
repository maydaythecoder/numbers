"""
Script to fix the model compatibility issue by rebuilding it with correct activations.
This loads the weights from the old model and saves a new compatible model.
"""
import os
import tensorflow as tf
import json

def fix_model():
    old_model_path = 'handwritten_digits.model.keras'
    backup_path = 'handwritten_digits.model.keras.backup'
    
    if not os.path.exists(old_model_path):
        print(f"Model file {old_model_path} not found!")
        return
    
    print("Backing up old model...")
    if os.path.exists(backup_path):
        os.remove(backup_path)
    os.rename(old_model_path, backup_path)
    
    print("Rebuilding model with correct architecture...")
    # Rebuild the model with string activations (Keras 3.x compatible)
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
    model.add(tf.keras.layers.Dense(units=128, activation='leaky_relu'))
    model.add(tf.keras.layers.Dense(units=128, activation='leaky_relu'))
    model.add(tf.keras.layers.Dense(units=10, activation='softmax'))
    
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
        print("You'll need to retrain the model. Run: python recognition.py")
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

