from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "handwritten_digits.model.keras"
DIGITS_DIR = PROJECT_ROOT / "digits"
