import sys
from itertools import count
from pathlib import Path

import cv2
import matplotlib.pyplot as plt

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from digit_recognition import (
    DIGITS_DIR,
    ensure_model,
    predict_preprocessed,
    preprocess_image_array,
)


def iter_digit_images():
    """Yield sequential digit image paths from the digits directory."""
    for image_number in count(1):
        image_path = DIGITS_DIR / f"digit{image_number}.png"
        if not image_path.is_file():
            break
        yield image_number, image_path


def load_digit_image(path):
    """Load a digit image as grayscale."""
    img_raw = cv2.imread(str(path))
    if img_raw is None:
        raise ValueError(f"Failed to load image {path.name}")
    return img_raw[:, :, 0]


def process_images():
    model = ensure_model()
    found_images = False

    for image_number, image_path in iter_digit_images():
        found_images = True
        try:
            img = load_digit_image(image_path)
            prepped = preprocess_image_array(img)
            digit, probabilities = predict_preprocessed(model, prepped)
            confidence = probabilities[digit] * 100

            print(
                f"{image_path.name}: predicted {digit} "
                f"({confidence:.1f}% confidence)"
            )

            plt.imshow(prepped[0], cmap="binary")
            plt.title(f"Predicted: {digit}")
            plt.axis("off")
            plt.show()
        except Exception as exc:  # pragma: no cover - keep processing other files
            print(f"Skipping {image_path.name}: {exc}")
            continue

    if not found_images:
        print(f"No digit images found in {DIGITS_DIR}")


if __name__ == "__main__":
    process_images()
