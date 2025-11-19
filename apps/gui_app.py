import sys
from pathlib import Path
import tkinter as tk
from tkinter import Canvas, Button, Label

import numpy as np
from PIL import Image, ImageDraw

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from digit_recognition import load_trained_model, predict_preprocessed, preprocess_image_array

class DigitRecognizerGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Handwritten Digit Recognition")
        
        # Load model
        try:
            self.model = load_trained_model()
        except Exception as e:
            raise RuntimeError(f"Could not load model. Error: {e}. You may need to retrain the model with: python scripts/recognition.py")
        
        # Canvas settings
        self.canvas_size = 250  # 25x25 grid * 10 pixels per cell
        self.grid_size = 25
        self.pixel_size = self.canvas_size // self.grid_size
        
        # Internal image buffer (mirrors canvas content for reliable capture)
        self.image_buffer = Image.new('RGB', (self.canvas_size, self.canvas_size), 'white')
        self.draw_buffer = ImageDraw.Draw(self.image_buffer)
        
        # Drawing state
        self.drawing = False
        self.last_x = None
        self.last_y = None
        
        # Create canvas (white background for black-on-white drawing, matching digit images)
        self.canvas = Canvas(root, width=self.canvas_size, height=self.canvas_size, 
                            bg='white', cursor='crosshair')
        self.canvas.pack(pady=10)
        
        # Bind mouse events
        self.canvas.bind('<Button-1>', self.start_draw)
        self.canvas.bind('<B1-Motion>', self.draw)
        self.canvas.bind('<ButtonRelease-1>', self.stop_draw)
        
        # Prediction label
        self.prediction_label = Label(root, text="Draw a digit", font=('Arial', 24))
        self.prediction_label.pack(pady=10)
        
        # Buttons
        button_frame = tk.Frame(root)
        button_frame.pack(pady=10)
        
        predict_btn = Button(button_frame, text="Predict", command=self.predict, 
                            font=('Arial', 14), bg='#4CAF50', fg='white', width=10)
        predict_btn.pack(side=tk.LEFT, padx=5)
        
        clear_btn = Button(button_frame, text="Clear", command=self.clear_canvas, 
                          font=('Arial', 14), bg='#f44336', fg='white', width=10)
        clear_btn.pack(side=tk.LEFT, padx=5)
        
    def start_draw(self, event):
        self.drawing = True
        self.last_x = event.x
        self.last_y = event.y
        # Draw initial point on canvas and buffer
        self.canvas.create_oval(event.x-4, event.y-4, event.x+4, event.y+4, 
                               fill='black', outline='black')
        self.draw_buffer.ellipse([event.x-4, event.y-4, event.x+4, event.y+4], 
                                fill='black', outline='black')
        
    def draw(self, event):
        if self.drawing:
            # Draw on canvas (black on white, matching digit images format)
            x, y = event.x, event.y
            if self.last_x and self.last_y:
                # Draw on Tkinter canvas
                self.canvas.create_line(self.last_x, self.last_y, x, y, 
                                       fill='black', width=8, capstyle=tk.ROUND, 
                                       smooth=tk.TRUE)
                # Mirror to internal image buffer
                self.draw_buffer.line([(self.last_x, self.last_y), (x, y)], 
                                     fill='black', width=8)
            else:
                # Draw on Tkinter canvas
                self.canvas.create_oval(x-4, y-4, x+4, y+4, fill='black', outline='black')
                # Mirror to internal image buffer
                self.draw_buffer.ellipse([x-4, y-4, x+4, y+4], fill='black', outline='black')
            self.last_x = x
            self.last_y = y
            
    def stop_draw(self, event):
        self.drawing = False
        self.last_x = None
        self.last_y = None
        
    def clear_canvas(self):
        self.canvas.delete("all")
        # Clear internal image buffer
        self.image_buffer = Image.new('RGB', (self.canvas_size, self.canvas_size), 'white')
        self.draw_buffer = ImageDraw.Draw(self.image_buffer)
        self.prediction_label.config(text="Draw a digit")
        
    def predict(self):
        # Use internal image buffer (matches web app approach - direct image capture)
        # Convert to grayscale (black on white, matching digit images format)
        img = self.image_buffer.convert('L')
        img = img.resize((28, 28), Image.Resampling.LANCZOS)
        
        # Convert to numpy array and preprocess
        # preprocess_image_array inverts (black on white -> white on black) and normalizes to [0, 1]
        img_array = np.array(img)
        img_array = preprocess_image_array(img_array)
        
        # Predict
        predicted_digit, probabilities = predict_preprocessed(self.model, img_array)
        confidence = probabilities[predicted_digit] * 100
        
        # Update label
        self.prediction_label.config(
            text=f"Prediction: {predicted_digit}\nConfidence: {confidence:.1f}%"
        )

if __name__ == "__main__":
    root = tk.Tk()
    app = DigitRecognizerGUI(root)
    root.mainloop()
