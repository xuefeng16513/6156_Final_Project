import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import to_categorical, plot_model
from tensorflow.keras.optimizers import Adam
import pandas as pd
import time
import psutil
import matplotlib.pyplot as plt

# Read training data
train_file_path = "train.csv"  
test_file_path = "test.csv" 

df_train = pd.read_csv(train_file_path)
df_test = pd.read_csv(test_file_path)

# Extract features and labels of training data
X_train = df_train.iloc[:, 1:].values
y_train = df_train.iloc[:, 0].values

# Extract features and labels for test data
X_test = df_test.iloc[:, 1:].values
y_test = df_test.iloc[:, 0].values

# Normalize pixel values
X_train = X_train / 255.0
X_test = X_test / 255.0

# Reshape to CNN input format (28, 28, 1)
X_train = X_train.reshape(-1, 28, 28, 1)
X_test = X_test.reshape(-1, 28, 28, 1)

num_classes = np.max(np.concatenate((y_train, y_test))) + 1
y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)

# Build CNN model
model = keras.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(num_classes, activation='softmax')
])

# Compile the model
# Set the learning rate
custom_adam = Adam(learning_rate=0.0001)
model.compile(optimizer=custom_adam,
              loss='categorical_crossentropy',
              metrics=['accuracy'])


# Train the model and record the training history
start_time = time.time()
history = model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test), batch_size=32)
total_training_time = time.time() - start_time
print(f"Total Training Time: {total_training_time:.2f} seconds")

# Evaluate the model
final_train_loss, final_train_acc = model.evaluate(X_train, y_train, verbose=0)
final_test_loss, final_test_acc = model.evaluate(X_test, y_test, verbose=0)

print(f"Final Training Loss: {final_train_loss:.4f}, Training Accuracy: {final_train_acc:.4f}")
print(f"Final Testing Loss: {final_test_loss:.4f}, Testing Accuracy: {final_test_acc:.4f}")

# Monitor CPU and memory usage
cpu_usage = psutil.cpu_percent(interval=1)
memory_info = psutil.virtual_memory()
print(f"CPU Usage: {cpu_usage}%")
print(f"RAM Usage: {memory_info.used / (1024**3):.2f} GB / {memory_info.total / (1024**3):.2f} GB")

# Test inference time
sample_input = np.random.rand(1, 28, 28, 1).astype(np.float32)
start_time = time.time()
_ = model.predict(sample_input)
end_time = time.time()
print(f"Inference Time per image: {(end_time - start_time) * 1000:.2f} ms")

# Draw loss and accuracy change curves
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
# plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('CNN Model Loss Over Epochs')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
# plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('CNN Model Accuracy Over Epochs')
plt.legend()

plt.show()

# Save the model
model.save("hand_sign_cnn_model.h5")
