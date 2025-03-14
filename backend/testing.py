import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import MeanSquaredError  # Import MSE

# Load the model once when the module is imported
import os
model_path = os.path.join(os.path.dirname(__file__), "utkface_model.h5")
model = load_model(model_path, custom_objects={"mse": MeanSquaredError()})

def analyze_image(image_path):
    print(image_path,"")
    """ Analyze image and return predicted age and gender """
    
    # Load and preprocess the image
    image = cv2.imread(image_path)
    if image is None:
        return {"error": "Invalid image or file path"}

    image = cv2.resize(image, (100, 100))  # Resize to match model input
    image = image / 255.0  # Normalize pixel values
    image = np.expand_dims(image, axis=0)  # Add batch dimension

    # Make prediction
    predictions = model.predict(image)

    # Extract predictions
    age_pred = predictions[0][0] * 100  # Adjust based on training normalization
    gender_pred = np.argmax(predictions[1])  # 0=Male, 1=Female

    return {
        "age": int(age_pred-5),
        "gender": "Male" if gender_pred == 0 else "Female"
    }
