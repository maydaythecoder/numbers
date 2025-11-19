import base64
import io
import re
import sys
from pathlib import Path

import numpy as np
from flask import Flask, jsonify, render_template, request
from PIL import Image

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from digit_recognition import load_trained_model, predict_preprocessed, preprocess_image_array

app = Flask(__name__, template_folder=str(PROJECT_ROOT / "templates"))

# Load model
try:
    model = load_trained_model()
except Exception as e:
    raise RuntimeError(f"Could not load model. Error: {e}. You may need to retrain the model with: python scripts/recognition.py")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get base64 image data from request
        data = request.json
        if data is None:
            return jsonify({'error': 'No JSON data provided'}), 400
        
        if 'image' not in data:
            return jsonify({'error': 'Missing "image" field in request'}), 400
        
        image_data = data['image']
        
        # Remove data URL prefix if present
        image_data = re.sub('^data:image/.+;base64,', '', image_data)
        
        # Decode base64 to image
        image_bytes = base64.b64decode(image_data)
        img = Image.open(io.BytesIO(image_bytes))
        
        # Convert to grayscale and resize to 28x28 (standard format)
        # Match the format of digit images in digits/ folder: black digits on white background
        img = img.convert('L')
        img = img.resize((28, 28), Image.Resampling.LANCZOS)
        
        # Convert to numpy array and preprocess
        # preprocess_image_array inverts (black on white -> white on black) and normalizes to [0, 1]
        img_array = np.array(img)
        prepped = preprocess_image_array(img_array)
        
        # Predict
        predicted_digit, probabilities = predict_preprocessed(model, prepped)
        confidence = float(probabilities[predicted_digit] * 100)
        probabilities = [float(p) * 100 for p in probabilities]
        
        return jsonify({
            'digit': predicted_digit,
            'confidence': confidence,
            'probabilities': probabilities
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, port=5000)
