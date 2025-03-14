import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split

# ✅ Define Dataset Path
DATASET_PATH = "backend/UTKFace/"

# ✅ Initialize Lists
images = []
genders = []

# ✅ Load Images and Extract Labels
for filename in os.listdir(DATASET_PATH):
    try:
        # Parse filename
        age, gender, race, _ = filename.split("_")
        gender = int(gender)  # 0 = Male, 1 = Female
        
        # Read and preprocess image
        img = cv2.imread(os.path.join(DATASET_PATH, filename))
        img = cv2.resize(img, (100, 100))  # Resize to 100x100
        img = img / 255.0  # Normalize pixels

        # Append to lists
        images.append(img)
        genders.append(gender)
    
    except Exception as e:
        print(f"Skipping {filename} due to error: {e}")

# ✅ Convert to NumPy Arrays
X = np.array(images)
y = np.array(genders)

# ✅ Train-Test Split (80-20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ✅ Convert labels to categorical format
y_train = to_categorical(y_train, num_classes=2)
y_test = to_categorical(y_test, num_classes=2)

print(f"Training samples: {X_train.shape[0]}, Testing samples: {X_test.shape[0]}")

# ✅ Define CNN Model
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(100, 100, 3)),
    MaxPooling2D(2,2),

    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),

    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2,2),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(2, activation='softmax')  # Softmax for binary classification (Male/Female)
])

# ✅ Compile Model
model.compile(optimizer=Adam(learning_rate=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# ✅ Train the Model
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=10,
    batch_size=32
)

# ✅ Save the Model
model.save("backend/utkface_gender_model.h5")
print("Model saved successfully!")

loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy * 100:.2f}%")