import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps
import os

class MLService:
    def __init__(self):
       
        model_path = os.path.join("saved_models", "pneumonia_model.h5")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at: {model_path}")
        
        self.model = tf.keras.models.load_model(model_path)
        print("Model loaded successfully!")

    def predict(self, image_bytes):
       
        img = Image.open(image_bytes).convert('L') # 'L' means Grayscale
        img = img.resize((150, 150))
        img_array = np.array(img) / 255.0
        img_array = img_array.reshape(-1, 150, 150, 1)
        
        # Inference
        prediction = self.model.predict(img_array)
        confidence_val = float(prediction[0][0])
        
       
        # Class 0 (Low value) -> Pneumonia
        # Class 1 (High value) -> Normal
        if confidence_val > 0.5:
            label = "Normal"
            probability = confidence_val * 100
        else:
            label = "Pneumonia"
            probability = (1 - confidence_val) * 100 
            
        return {
            "prediction": label,
            "confidence": f"{round(probability, 2)}%"
        }