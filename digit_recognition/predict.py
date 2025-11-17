import numpy as np

from .preprocess import preprocess_image_array


def predict_preprocessed(model, preprocessed_batch):
    """Run prediction for a batch already shaped and normalized for the model."""
    probabilities = model.predict(preprocessed_batch, verbose=0)[0]
    digit = int(np.argmax(probabilities))
    return digit, probabilities


def predict_digit(model, img_array, **preprocess_kwargs):
    """Preprocess an array/image and return the predicted digit and probabilities."""
    prepped = preprocess_image_array(img_array, **preprocess_kwargs)
    return predict_preprocessed(model, prepped)
