import cv2
import numpy as np
import tensorflow as tf
import time
import os
import mediapipe as mp # Import MediaPipe
from collections import deque


# --- Constants ---
MODEL_PATH = "hand_sign_cnn_model.h5"
IMG_SIZE = 28  # Input size expected by the CNN model (MUST MATCH TRAINING)
last_predictions = deque(maxlen=5)  # Save the last 5 frames of prediction
# ROI constants are no longer needed for the fixed box

# --- MediaPipe Hand Detection Setup ---
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
# Initialize MediaPipe Hands
# - static_image_mode=False: Process video stream
# - max_num_hands=1: We only care about one hand for ASL letters
# - min_detection_confidence=0.7: Adjust if hands aren't detected well
# - min_tracking_confidence=0.5: Adjust if tracking is lost easily
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5)

# --- Letter Mapping (Corrected for 25 classes based on previous output) ---
# Based on your evaluate script detecting 25 classes (indices 0-24)
# Assuming standard Sign Language MNIST mapping (A-Z, skipping J)
index_to_letter = {
     0:'A', 1:'B', 2:'C', 3:'D', 4:'E', 5:'F', 6:'G', 7:'H', 8:'I',
     # Index 9 (J) is skipped in the standard SL-MNIST dataset label scheme
     10:'K', 11:'L', 12:'M', 13:'N', 14:'O', 15:'P', 16:'Q', 17:'R',
     18:'S', 19:'T', 20:'U', 21:'V', 22:'W', 23:'X', 24:'Y', 25:'Z'
     # This mapping assumes index 25 is Z, matching 26 total letters mapped
     # across the available indices 0-24 (if J is truly skipped). Let's assume 0-24 = A-Y (no Z) to match the 25 classes
}
index_to_letter = {
     0:'A', 1:'B', 2:'C', 3:'D', 4:'E', 5:'F', 6:'G', 7:'H', 8:'I',
     9:'K', 10:'L', 11:'M', 12:'N', 13:'O', 14:'P', 15:'Q', 16:'R',
    17:'S', 18:'T', 19:'U', 20:'V', 21:'W', 22:'X', 23:'Y', 24:'Z' # Assuming 0-24 maps to A-Z skipping J
}
print(f"Using updated label mapping for 25 classes (Indices 0-24). Assuming A-Z skipping J.")


# --- Load the Trained Model ---
print(f"Loading model from '{MODEL_PATH}'...")
if not os.path.exists(MODEL_PATH):
    print(f"Error: Model file not found at '{MODEL_PATH}'")
    print("Please run the training script first to create this file.")
    exit()
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

# --- Initialize Camera ---
cap = cv2.VideoCapture(0) # Use 0 for the default camera
if not cap.isOpened():
    print("Error: Could not open camera. Check if it's connected and not used by another app.")
    exit()

print("Camera initialized. Position your hand in view.")
print("Press 'q' to quit.")

# --- Real-time Recognition Loop ---
last_prediction_time = time.time()
prediction_interval = 0.2 # Predict roughly 5 times per second - adjust if needed
predicted_text = "Initializing..."
bbox_padding = 25 # Pixels to add around the detected hand bounding box

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame.")
        time.sleep(1)
        continue

    # Flip the frame horizontally (mirror effect)
    frame = cv2.flip(frame, 1)
    display_frame = frame.copy() # Create a copy for drawing landmarks etc.
    frame_height, frame_width, _ = frame.shape

    # --- MediaPipe Hand Detection ---
    # Convert frame to RGB (MediaPipe uses RGB)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # Process the frame to find hands
    results = hands.process(frame_rgb)

    # Default text if no hand is detected
    current_predicted_text = "No Hand Detected"

    if results.multi_hand_landmarks:
        # Process only the first detected hand
        hand_landmarks = results.multi_hand_landmarks[0]

        # Draw hand landmarks on the display frame (optional but helpful)
        mp_drawing.draw_landmarks(
            display_frame,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS)

        # --- Calculate Bounding Box around the hand ---
        x_coords = [landmark.x for landmark in hand_landmarks.landmark]
        y_coords = [landmark.y for landmark in hand_landmarks.landmark]

        # Get pixel coordinates, add padding, and ensure they are within frame bounds
        x_min = max(0, int(min(x_coords) * frame_width) - bbox_padding)
        y_min = max(0, int(min(y_coords) * frame_height) - bbox_padding)
        x_max = min(frame_width, int(max(x_coords) * frame_width) + bbox_padding)
        y_max = min(frame_height, int(max(y_coords) * frame_height) + bbox_padding)

        # Draw bounding box on the display frame (optional)
        cv2.rectangle(display_frame, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)

        # --- Prediction Logic (Run periodically) ---
        current_time = time.time()
        if current_time - last_prediction_time >= prediction_interval:
            last_prediction_time = current_time

            # Extract the hand ROI from the *original* frame using the calculated bbox
            roi = frame[y_min:y_max, x_min:x_max]

            # Ensure ROI is valid
            if roi.size > 0:
                # --- Preprocess the Cropped Hand ROI ---
                try:
                    # 1. Convert to Grayscale
                    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                    # 2. Resize to the model's expected input size (28x28)
                    resized_roi = cv2.resize(gray_roi, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)

                    zoom_factor = 10
                    zoomed_roi = cv2.resize(resized_roi, (IMG_SIZE * zoom_factor, IMG_SIZE * zoom_factor), interpolation=cv2.INTER_NEAREST)
                    cv2.imshow('Preprocessed Input (Zoomed)', zoomed_roi)

                    # 3. Normalize pixel values (0-1)
                    normalized_roi = resized_roi / 255.0
                    # 4. Reshape for the model (batch, height, width, channels)
                    input_data = normalized_roi.reshape(1, IMG_SIZE, IMG_SIZE, 1)

                    # --- Make Prediction ---
                    prediction = model.predict(input_data, verbose=0)
                    predicted_index = np.argmax(prediction[0])
                    confidence = np.max(prediction[0]) * 100
                    # predicted_letter = index_to_letter.get(predicted_index, '?')
                    last_predictions.append(predicted_index)
                    voted_index = max(set(last_predictions), key=last_predictions.count)
                    predicted_letter = index_to_letter.get(voted_index, '?')


                    # Update the text to display
                    predicted_text = f"{predicted_letter} ({confidence:.1f}%)"
                    current_predicted_text = predicted_text # Update text for this frame

                except Exception as e:
                    print(f"Error during processing/prediction: {e}")
                    predicted_text = "Processing Error"
                    current_predicted_text = predicted_text
            else:
                predicted_text = "ROI Error (Hand too close to edge?)"
                current_predicted_text = predicted_text
        else:
             # Keep displaying the last prediction if interval hasn't passed
             current_predicted_text = predicted_text
    else:
        # If no hand landmarks detected, use the default text
        predicted_text = "No Hand Detected" # Reset persistent text
        current_predicted_text = predicted_text


    # --- Display the Prediction Text on the frame ---
    cv2.putText(display_frame, current_predicted_text, (10, 30), # Position text top-left
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('ASL Letter Recognition (MediaPipe - Press Q to Quit)', display_frame)

    # --- Exit Condition ---
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# --- Cleanup ---
print("Releasing resources...")
cap.release()
hands.close() # Close MediaPipe Hands
cv2.destroyAllWindows()
print("Application exited.")
