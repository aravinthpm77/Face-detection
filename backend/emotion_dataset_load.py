import tensorflow as tf

# Define paths
train_dir = "backend/archive/train"
test_dir = "backend/archive/test"

# Load dataset
train_data = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    image_size=(48, 48),  # FER-2013 image size
    batch_size=32
)

test_data = tf.keras.utils.image_dataset_from_directory(
    test_dir,
    image_size=(48, 48),
    batch_size=32
)

# Check class names
class_names = train_data.class_names
print("Emotion Classes:", class_names)
