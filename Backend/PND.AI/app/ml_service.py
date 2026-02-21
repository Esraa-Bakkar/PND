import tensorflow as tf
import numpy as np
from PIL import Image
import os
import json

class MLService:
    def __init__(self):
        
        config_path = os.path.join("saved_models", "models_config.json")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found at: {config_path}")

        with open(config_path, 'r') as f:
            self.full_config = json.load(f)
        
        # Active Model
        self.active_key = self.full_config.get("active_model", "densenet_v1")
        self.config = self.full_config["models"][self.active_key]
        
      
        model_full_path = os.path.join("saved_models", self.config["model_name"])
        if not os.path.exists(model_full_path):
            raise FileNotFoundError(f"Model file {self.config['model_name']} not found!")
            
        self.model = tf.keras.models.load_model(model_full_path)
        print(f"--- ML Engine Started ---")
        print(f"Active Registry Key: {self.active_key}")
        print(f"Model File: {self.config['model_name']}")
        print(f"Target Size: {self.config['target_size']}")
        print(f"-------------------------")

    def predict(self, image_bytes):
     
        img = Image.open(image_bytes).convert(self.config["color_mode"])
        
        #  Resize
        target_size = tuple(self.config["target_size"])
        img = img.resize(target_size)
        
        #  Normalization
        img_array = np.array(img) / self.config["normalization_factor"]
        
        # Reshape
        img_array = img_array.reshape(-1, target_size[0], target_size[1], self.config["reshape_channels"])
        
        #  Inference
        prediction = self.model.predict(img_array)
        confidence_val = float(prediction[0][0])
        
        # Threshold 
        if confidence_val > self.config["threshold"]:
            label = self.config["labels"]["above_threshold"]
            probability = confidence_val * 100
        else:
            label = self.config["labels"]["below_threshold"]
            probability = (1 - confidence_val) * 100
            
        return {
            "prediction": label,
            "confidence": f"{round(probability, 2)}%",
            "metadata": {
                "active_model": self.active_key,
                "model_file": self.config["model_name"]
            }
        }