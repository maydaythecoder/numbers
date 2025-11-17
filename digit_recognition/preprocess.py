from typing import Optional, Tuple

import numpy as np
from PIL import Image

ImageSize = Optional[Tuple[int, int]]


def preprocess_image_array(
    img_array,
    target_size: ImageSize = (28, 28),
    invert: bool = True,
):
    """
    Normalize and shape an image array for the model.

    Accepts numpy arrays or PIL Images. Converts to grayscale, resizes if needed,
    optionally inverts, normalizes to [0, 1], and returns shape (1, H, W).
    """
    arr = np.array(img_array)

    # Drop color channels if present
    if arr.ndim == 3:
        arr = arr[:, :, 0]

    if target_size and arr.shape != target_size:
        arr = np.array(Image.fromarray(arr).resize(target_size))

    if invert:
        arr = np.invert(arr)

    arr = arr.astype("float32")
    if arr.max() > 1.0:
        arr /= 255.0

    if target_size:
        height, width = target_size
    else:
        height, width = arr.shape

    return arr.reshape(1, height, width)
