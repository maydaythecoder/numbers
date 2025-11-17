import tkinter as tk
from tkinter import Canvas, Button, Label
import numpy as np
import tensorflow as tf
from PIL import Image, ImageDraw
import io

class DigitRecognizerGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Handwritten Digit Recognition")
        
        # Load model
        try:
            self.model = tf.keras.models.load_model('handwritten_digits.model.keras')
        except Exception as e:
            raise RuntimeError(f"Could not load model. Error: {e}. You may need to retrain the model with: python recognition.py")
        
        # Canvas settings
        self.canvas_size = 250  # 25x25 grid * 10 pixels per cell
        self.grid_size = 25
        self.pixel_size = self.canvas_size // self.grid_size
        
        # Drawing state
        self.drawing = False
        self.last_x = None
        self.last_y = None
        
        # Create canvas
        self.canvas = Canvas(root, width=self.canvas_size, height=self.canvas_size, 
                            bg='black', cursor='crosshair')
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
        
    def draw(self, event):
        if self.drawing:
            # Draw on canvas
            x, y = event.x, event.y
            if self.last_x and self.last_y:
                self.canvas.create_line(self.last_x, self.last_y, x, y, 
                                       fill='white', width=8, capstyle=tk.ROUND, 
                                       smooth=tk.TRUE)
            else:
                self.canvas.create_oval(x-4, y-4, x+4, y+4, fill='white', outline='white')
            self.last_x = x
            self.last_y = y
            
    def stop_draw(self, event):
        self.drawing = False
        self.last_x = None
        self.last_y = None
        
    def clear_canvas(self):
        self.canvas.delete("all")
        self.prediction_label.config(text="Draw a digit")
        
    def predict(self):
        # Get canvas content as image
        ps = self.canvas.postscript(colormode='mono')
        img = Image.open(io.BytesIO(ps.encode('utf-8')))
        
        # Convert to grayscale array
        img = img.convert('L')
        img = img.resize((28, 28), Image.Resampling.LANCZOS)
        
        # Convert to numpy array and normalize
        img_array = np.array(img)
        img_array = np.invert(img_array)  # Invert colors (MNIST is white on black)
        img_array = img_array.astype('float32') / 255.0
        
        # Reshape for model (1, 28, 28)
        img_array = img_array.reshape(1, 28, 28)
        
        # Predict
        prediction = self.model.predict(img_array, verbose=0)
        predicted_digit = np.argmax(prediction)
        confidence = prediction[0][predicted_digit] * 100
        
        # Update label
        self.prediction_label.config(
            text=f"Prediction: {predicted_digit}\nConfidence: {confidence:.1f}%"
        )

if __name__ == "__main__":
    root = tk.Tk()
    app = DigitRecognizerGUI(root)
    root.mainloop()