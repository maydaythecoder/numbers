from pathlib import Path
from typing import Tuple

import tensorflow as tf

from .data import load_mnist_data
from .paths import MODEL_PATH

DEFAULT_EPOCHS = 3


def build_model():
    """Create the classifier architecture."""
    model = tf.keras.models.Sequential(name="digit_classifier")
    model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
    model.add(tf.keras.layers.Dense(units=128, activation="leaky_relu"))
    model.add(tf.keras.layers.Dense(units=128, activation="leaky_relu"))
    model.add(tf.keras.layers.Dense(units=10, activation="softmax"))
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def train_model(
    epochs: int = DEFAULT_EPOCHS,
    save_path: Path = MODEL_PATH,
    verbose: int = 1,
) -> Tuple[tf.keras.Model, Tuple[float, float]]:
    """Train the model on MNIST and save it."""
    (x_train, y_train), (x_test, y_test) = load_mnist_data()
    model = build_model()
    model.fit(x_train, y_train, epochs=epochs, verbose=verbose)
    val_loss, val_acc = model.evaluate(x_test, y_test, verbose=0)
    model.save(save_path)
    return model, (float(val_loss), float(val_acc))


def load_trained_model(path: Path = MODEL_PATH) -> tf.keras.Model:
    """Load a saved model, handling safe_mode compatibility."""
    if not path.exists():
        raise FileNotFoundError(f"Model file not found at {path}")
    try:
        return tf.keras.models.load_model(path, safe_mode=False)
    except Exception:
        return tf.keras.models.load_model(path)


def ensure_model(
    path: Path = MODEL_PATH,
    epochs: int = DEFAULT_EPOCHS,
) -> tf.keras.Model:
    """Load an existing model or train a new one if missing."""
    if path.exists():
        return load_trained_model(path)
    model, (val_loss, val_acc) = train_model(epochs=epochs, save_path=path)
    print(f"Trained new model. Validation loss: {val_loss:.4f}, acc: {val_acc:.4f}")
    return model
