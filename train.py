import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import random

# Set random seeds for reproducibility
seed = 42
tf.random.set_seed(seed)
np.random.seed(seed)
random.seed(seed)

# Define your Hindi class labels
class_labels = ["1", "2", "3", "4", "5", "6", "7", "8"]  # Add all your Hindi literals here

# Create a dictionary to map Hindi class labels to integer labels
label_to_int = {label: idx for idx, label in enumerate(class_labels)}

# Function to load all image paths and their corresponding labels from the dataset directory
def load_image_paths_and_labels(data_dir):
    image_paths = []
    labels = []
    for label_dir in os.listdir(data_dir):
        label_path = os.path.join(data_dir, label_dir)
        if os.path.isdir(label_path):
            for img_file in os.listdir(label_path):
                img_path = os.path.join(label_path, img_file)
                image_paths.append(img_path)
                labels.append(label_to_int[label_dir])  # Use the mapping to convert label to integer
    return image_paths, np.array(labels)

# Path to your dataset directory (replace with your actual path)
data_dir = 'data'

# Load all image paths and labels
image_paths, labels = load_image_paths_and_labels(data_dir)

# Function to preprocess images
def load_and_preprocess_images(image_paths):
    def preprocess(path):
        image = tf.io.read_file(path)
        image = tf.image.decode_image(image, channels=3)
        image = tf.image.resize(image, [64, 64])
        image = image / 255.0  # Normalize to [0, 1]
        return image

    images = [preprocess(path) for path in image_paths]
    return tf.stack(images)

# Preprocess all images
batch_images = load_and_preprocess_images(image_paths)

# Convert TensorFlow tensor to NumPy array before splitting
batch_images_np = batch_images.numpy()

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(batch_images_np, labels, test_size=0.2, random_state=42)

# Define the CNN model
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(64, 64, 3)),  # Input shape for 64x64 RGB images
    tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(len(class_labels), activation='softmax')  # Adjust the number of classes based on Hindi literals
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

# Save the trained model
model.save('check.h5')
