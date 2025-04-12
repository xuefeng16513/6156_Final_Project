import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
import pandas as pd
import os # Import os to check if file exists

# --- Configuration ---
MODEL_PATH = "hand_sign_cnn_model.h5"  # Path to your saved model
TEST_DATA_PATH = "test.csv"           # Path to your test data CSV

# --- Load Test Data ---
# Check if test data file exists
if not os.path.exists(TEST_DATA_PATH):
    print(f"Error: Test data file not found at '{TEST_DATA_PATH}'")
    print("Please download the Sign Language MNIST dataset (e.g., from Kaggle) and place test.csv here.")
    exit()

print(f"Loading test data from '{TEST_DATA_PATH}'...")
try:
    df_test = pd.read_csv(TEST_DATA_PATH)
except Exception as e:
    print(f"Error reading CSV file: {e}")
    exit()

# --- Prepare Test Data ---
print("Preparing test data...")
# Extract features (pixels) and labels
X_test = df_test.iloc[:, 1:].values
y_test_labels = df_test.iloc[:, 0].values # Original numerical labels

# Normalize pixel values (0-1)
X_test = X_test / 255.0

# Reshape to CNN input format (batch, height, width, channels) -> (batch, 28, 28, 1)
X_test = X_test.reshape(-1, 28, 28, 1)

# Determine the number of classes from the test labels
num_classes = np.max(y_test_labels) + 1
print(f"Detected {num_classes} classes based on test data.")

# One-hot encode the labels (important for 'categorical_crossentropy' loss)
y_test_categorical = to_categorical(y_test_labels, num_classes)

print("Test data prepared.")
print(f"Test data shape: {X_test.shape}")
print(f"Test labels shape: {y_test_categorical.shape}")


# --- Load the Pre-trained Model ---
print(f"Loading model from '{MODEL_PATH}'...")
# Check if model file exists
if not os.path.exists(MODEL_PATH):
    print(f"Error: Model file not found at '{MODEL_PATH}'")
    print("Please run the training script first to create this file.")
    exit()

try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print("Model loaded successfully!")
    model.summary() # Print model structure
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

# --- Evaluate the Loaded Model ---
print("Evaluating model performance on the test set...")
# The model needs to be compiled before evaluation if loaded this way,
# but load_model usually handles this if the optimizer state was saved.
# If you encounter issues, uncomment the compilation line:
# model.compile(loss='categorical_crossentropy', metrics=['accuracy'])

try:
    loss, accuracy = model.evaluate(X_test, y_test_categorical, verbose=1)
    print("\nEvaluation Results:")
    print(f"Test Loss: {loss:.4f}")
    print(f"Test Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")
except Exception as e:
    print(f"Error during model evaluation: {e}")

print("\nEvaluation script finished.")