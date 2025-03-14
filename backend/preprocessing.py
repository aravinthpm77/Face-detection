import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.utils import to_categorical

# 🔹 Image size for the model
IMG_SIZE = 100

# 🔹 Data storage
X, age_labels, gender_labels = [], [], []

# 🔹 Load UTKFace Dataset
data_dir = "backend/UTKFace/"  # Update this path
for file in os.listdir(data_dir):
    try:
        # 🔥 Fix: Extract only age and gender from the filename
        parts = file.split("_")

        if len(parts) < 2:
            print(f"Skipping {file}: Unexpected filename format")
            continue

        age = int(parts[0])
        gender = int(parts[1])

        # 🔥 Remove incorrect values
        if age < 1 or age > 100:
            print(f"Skipping {file}: Unreasonable age value")
            continue

        if gender not in [0, 1]:  # Ensure gender is 0 (Female) or 1 (Male)
            print(f"Skipping {file}: Invalid gender value {gender}")
            continue

        # 🔹 Read and preprocess image
        img = cv2.imread(os.path.join(data_dir, file))
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img = img / 255.0  # Normalize

        # 🔹 Append to dataset
        X.append(img)
        age_labels.append(age / 100)  # Normalize age (0-1 range)
        gender_labels.append(gender)

    except Exception as e:
        print(f"Skipping {file}: {e}")

# 🔹 Convert to NumPy arrays
X = np.array(X)
age_labels = np.array(age_labels)
gender_labels = np.array(gender_labels)

# 🔹 One-Hot Encode Gender (0 = Female, 1 = Male)
gender_labels = to_categorical(gender_labels, num_classes=2)

# 🔹 Save Preprocessed Data
np.savez_compressed("preprocessed_data.npz", X=X, age_labels=age_labels, gender_labels=gender_labels)
print("✅ Preprocessed data saved successfully!")
