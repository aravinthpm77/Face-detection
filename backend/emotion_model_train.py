import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import os

# 📂 Define dataset paths
train_dir = "backend/archive/train"  # 🔄 Change this to your dataset path
test_dir = "backend/archive/test"

# 📏 Image size & batch settings
IMG_SIZE = (48, 48)  
BATCH_SIZE = 32

# 🔄 Load dataset from folders
train_data = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int"
)

test_data = tf.keras.utils.image_dataset_from_directory(
    test_dir,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int"
)

# 🎭 Emotion classes
class_names = train_data.class_names
print("Emotion Classes:", class_names)

# 🔄 Normalize pixel values (0-255 → 0-1)
normalization_layer = layers.Rescaling(1./255)

# 🎨 Data Augmentation
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1)
])

# 🛠 Apply augmentation & normalization
train_data = train_data.map(lambda x, y: (normalization_layer(data_augmentation(x)), y))
test_data = test_data.map(lambda x, y: (normalization_layer(x), y))

# 🚀 Build CNN Model
model = models.Sequential([
    layers.Conv2D(64, (3, 3), activation="relu", input_shape=(48, 48, 3)),
    layers.MaxPooling2D(2, 2),

    layers.Conv2D(128, (3, 3), activation="relu"),
    layers.MaxPooling2D(2, 2),

    layers.Conv2D(256, (3, 3), activation="relu"),
    layers.MaxPooling2D(2, 2),

    layers.Flatten(),
    layers.Dense(512, activation="relu"),
    layers.Dropout(0.5),
    layers.Dense(len(class_names), activation="softmax")  # 🧑‍🤖 7 emotion classes
])

# ⚙️ Compile Model
model.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

# 🏗 Model Summary
model.summary()

# 🎯 Train the Model
EPOCHS = 20  # Adjust based on your hardware
history = model.fit(
    train_data,
    validation_data=test_data,
    epochs=EPOCHS
)

# 📈 Plot Training Performance
plt.figure(figsize=(12, 5))

# 📊 Accuracy Plot
plt.subplot(1, 2, 1)
plt.plot(history.history["accuracy"], label="Train Accuracy")
plt.plot(history.history["val_accuracy"], label="Val Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.title("Model Accuracy")

# 📉 Loss Plot
plt.subplot(1, 2, 2)
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Val Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.title("Model Loss")

plt.show()

# 💾 Save the Model
model.save("backend/emotion_detection_model.h5")
print("🎉 Model saved successfully!")
