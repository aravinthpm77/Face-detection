import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
from sklearn.model_selection import train_test_split

# 🔹 Load Preprocessed Data
data = np.load("backend/preprocessed_data.npz")
X, age_labels, gender_labels = data["X"], data["age_labels"], data["gender_labels"]

# 🔹 Split Data (Train/Test)
X_train, X_test, age_train, age_test, gender_train, gender_test = train_test_split(
    X, age_labels, gender_labels, test_size=0.2, random_state=42
)

# 🔹 Build Optimized Model
def build_model():
    inputs = Input(shape=(100, 100, 3))  # Explicit Input Layer

    x = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = BatchNormalization()(x)
    x = MaxPooling2D(2, 2)(x)

    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(2, 2)(x)

    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(2, 2)(x)

    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)

    # 🔹 Age Prediction (Regression)
    age_output = Dense(1, activation="linear", name="age_output")(x)

    # 🔹 Gender Prediction (Classification)
    gender_output = Dense(2, activation="softmax", name="gender_output")(x)

    return Model(inputs=inputs, outputs=[age_output, gender_output])

# 🔹 Compile Model
model = build_model()
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss={"age_output": MeanSquaredError(), "gender_output": "categorical_crossentropy"},
    metrics={"age_output": "mae", "gender_output": "accuracy"}
)

# 🔹 Train Model
history = model.fit(
    X_train, {"age_output": age_train, "gender_output": gender_train},
    validation_data=(X_test, {"age_output": age_test, "gender_output": gender_test}),
    epochs=50,  # 🔥 Increase epochs for better learning
    batch_size=32
)

# 🔹 Save Model
model.save("backend/face_model.h5")
print("✅ Model trained and saved as 'face_model.h5'")
