from flask import Flask, render_template, request, jsonify
import numpy as np
import tensorflow as tf
from PIL import Image
import base64
import io
import re

app = Flask(__name__)

# Load model
try:
    model = tf.keras.models.load_model('handwritten_digits.model.keras')
except Exception as e:
    raise RuntimeError(f"Could not load model. Error: {e}. You may need to retrain the model with: python recognition.py")

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
        
        # Convert to grayscale and resize to 28x28
        img = img.convert('L')
        img = img.resize((28, 28), Image.Resampling.LANCZOS)
        
        # Convert to numpy array and normalize
        img_array = np.array(img)
        img_array = np.invert(img_array)  # Invert colors
        img_array = img_array.astype('float32') / 255.0
        
        # Reshape for model
        img_array = img_array.reshape(1, 28, 28)
        
        # Predict
        prediction = model.predict(img_array, verbose=0)
        predicted_digit = int(np.argmax(prediction))
        confidence = float(prediction[0][predicted_digit] * 100)
        
        # Get all probabilities
        probabilities = [float(p) * 100 for p in prediction[0]]
        
        return jsonify({
            'digit': predicted_digit,
            'confidence': confidence,
            'probabilities': probabilities
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, port=5000)