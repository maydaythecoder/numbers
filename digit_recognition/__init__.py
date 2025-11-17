"""Shared utilities for the handwritten digit recognition project."""

from .model import (
    build_model,
    ensure_model,
    load_trained_model,
    train_model,
)
from .paths import DIGITS_DIR, MODEL_PATH, PROJECT_ROOT
from .predict import predict_digit, predict_preprocessed
from .preprocess import preprocess_image_array

__all__ = [
    "build_model",
    "ensure_model",
    "load_trained_model",
    "train_model",
    "DIGITS_DIR",
    "MODEL_PATH",
    "PROJECT_ROOT",
    "predict_digit",
    "predict_preprocessed",
    "preprocess_image_array",
]
