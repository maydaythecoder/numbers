# Handwritten Digit Recognition

A neural network-based handwritten digit recognition system using TensorFlow/Keras. The model is trained on the MNIST dataset and can recognize digits from image files.

## Features

- **Multiple Interfaces:**
  - 🌐 Web interface with interactive drawing canvas
  - 🖥️ Desktop GUI application
  - 📸 Batch image processing from files
- Automatic model training (if no saved model exists)
- Real-time digit recognition with confidence scores
- Uses a deep neural network with 2 hidden layers (128 units each)

## Requirements

- Python 3.9 or higher
- Virtual environment (venv)

## Setup

1. **Clone or navigate to the project directory:**

   ```bash
   cd numbers
   ```

2. **Create a virtual environment:**

   ```bash
   python -m venv venv
   ```

3. **Activate the virtual environment:**

   On macOS/Linux:

   ```bash
   source venv/bin/activate
   ```

   On Windows:

   ```bash
   venv\Scripts\activate
   ```

4. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Quick Start (Recommended)

Use the unified launcher script:

```bash
# Interactive menu
python run.py

# Or directly specify the mode
python run.py web      # Start web interface
python run.py gui      # Start desktop GUI
python run.py images   # Process images from digits/ folder
```

You can also use the shell script:

```bash
./run.sh web    # Web interface
./run.sh gui    # Desktop GUI
./run.sh images # Process images
```

### Application Modes

#### 1. Web Interface

```bash
python run.py web
# or
python web_app.py
```

- Opens a web server on `http://localhost:5000`
- Draw digits in a 25x25 grid canvas
- View predictions with confidence scores and probability distribution
- Works in any modern web browser

#### 2. Desktop GUI

```bash
python run.py gui
# or
python gui_app.py
```

- Native desktop application
- Draw digits with your mouse
- Real-time predictions
- Simple and fast

#### 3. Batch Image Processing

```bash
python run.py images
# or
python recognition.py
```

1. **Place your digit images in the `digits/` folder:**
   - Name them as `digit1.png`, `digit2.png`, `digit3.png`, etc.
   - Images should be grayscale digit images

2. **What happens:**
   - **First run:** The script will automatically download the MNIST dataset, train a new model (takes a few minutes), save it as `handwritten_digits.model.keras`, then process your images
   - **Subsequent runs:** The script will load the saved model and immediately process your images

3. **Output:**
   - The script prints predictions for each digit image
   - Each image is displayed using matplotlib
   - Close each image window to proceed to the next one

## Project Structure

```text
numbers/
├── digits/                  # Place your digit images here (digit1.png, digit2.png, ...)
├── templates/
│   └── index.html          # Web interface HTML template
├── recognition.py           # Batch image processing script
├── web_app.py              # Flask web application
├── gui_app.py              # Tkinter desktop GUI
├── run.py                  # Unified launcher script (recommended)
├── run.sh                  # Shell launcher script
├── requirements.txt        # Python dependencies
├── pyrightconfig.json      # Type checker configuration
├── tensorflow.pyi          # Type stubs for TensorFlow
└── README.md              # This file
```

## Model Details

- **Architecture:** Sequential neural network
  - Input layer: Flattened 28x28 images
  - Hidden layer 1: 128 units with LeakyReLU activation
  - Hidden layer 2: 128 units with LeakyReLU activation
  - Output layer: 10 units (digits 0-9) with softmax activation
- **Training:** 3 epochs on MNIST dataset
- **Expected accuracy:** ~97% on test set

## Notes

- The model automatically trains on first run if `handwritten_digits.model.keras` doesn't exist
- Images should be in PNG format and named sequentially starting from `digit1.png`
- The script processes images until it can't find the next sequential image
