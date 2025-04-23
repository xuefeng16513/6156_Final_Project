import cv2
import numpy as np
import tensorflow as tf
import time
import os
import mediapipe as mp # Import MediaPipe
from collections import deque

# --- Constants ---
MODEL_PATH = "cnn_mlp_generator_model.h5"
IMG_SIZE = 128  # Input size expected by the CNN model (MUST MATCH TRAINING)
last_predictions = deque(maxlen=5)  # Save the last 5 frames of prediction
# ROI constants are no longer needed for the fixed box
recognized_text = ""  # Save the recognized complete string
last_append_time = time.time()

def resize_with_padding(img, size=128):
    h, w = img.shape[:2]
    scale = size / max(h, w)
    nh, nw = int(h * scale), int(w * scale)
    img_resized = cv2.resize(img, (nw, nh))

    # create background
    pad_top = (size - nh) // 2
    pad_bottom = size - nh - pad_top
    pad_left = (size - nw) // 2
    pad_right = size - nw - pad_left

    img_padded = cv2.copyMakeBorder(img_resized, pad_top, pad_bottom, pad_left, pad_right,
                                    borderType=cv2.BORDER_CONSTANT, value=[0, 0, 0])
    return img_padded

def extract_123d_keypoints(landmarks):
    coords = np.array([[lm.x, lm.y, lm.z] for lm in landmarks])
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 4),
        (0, 5), (5, 6), (6, 7), (7, 8),
        (0, 9), (9,10), (10,11), (11,12),
        (0,13), (13,14), (14,15), (15,16),
        (0,17), (17,18), (18,19), (19,20)
    ]
    diffs = [coords[j] - coords[i] for i, j in edges]
    flat_kp = np.concatenate([coords.flatten()] + [np.array(diffs).flatten()])
    return flat_kp

def render_text_window(text, width=640, height=460, line_height=40, font_scale=1.2):
    window = np.ones((height, width, 3), dtype=np.uint8) * 255

    lines = []
    current_line = ''
    for ch in text:
        test_line = current_line + ch
        text_size = cv2.getTextSize(test_line, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)[0]
        if text_size[0] < width - 20:
            current_line = test_line
        else:
            lines.append(current_line)
            current_line = ch
    if current_line:
        lines.append(current_line)

    # Automatic scrolling
    max_lines = height // line_height
    lines_to_display = lines[-max_lines:]

    for i, line in enumerate(lines_to_display):
        y = (i + 1) * line_height
        cv2.putText(window, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), 2)

    return window

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
     18:'S', 19:'T', 20:'U', 21:'V', 22:'W', 23:'X', 24:'Y', 25:'Z',26: 'del', 27: 'nothing', 28: 'space'
     # This mapping assumes index 25 is Z, matching 26 total letters mapped
     # across the available indices 0-24 (if J is truly skipped). Let's assume 0-24 = A-Y (no Z) to match the 25 classes
}
# index_to_letter = {
#     0:'A', 1:'B', 2:'C', 3:'D', 4:'E', 5:'F', 6:'G', 7:'H', 8:'I',
#     9:'K', 10:'L', 11:'M', 12:'N', 13:'O', 14:'P', 15:'Q', 16:'R',
#    17:'S', 18:'T', 19:'U', 20:'V', 21:'W', 22:'X', 23:'Y', 24:'Z' # Assuming 0-24 maps to A-Z skipping J
# }
# print(f"Using updated label mapping for 25 classes (Indices 0-24). Assuming A-Z skipping J.")


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
                    # gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                    # 2. Resize to the model's expected input size (28x28)
                    # resized_roi = cv2.resize(gray_roi, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
                    # resized_roi = cv2.resize(roi, (IMG_SIZE, IMG_SIZE))
                    resized_roi = resize_with_padding(roi, IMG_SIZE)

                    zoom_factor = 2
                    zoomed_roi = cv2.resize(resized_roi, (IMG_SIZE * zoom_factor, IMG_SIZE * zoom_factor), interpolation=cv2.INTER_NEAREST)
                    cv2.imshow('Preprocessed Input (Zoomed)', zoomed_roi)

                    # 3. Normalize pixel values (0-1)
                    normalized_roi = resized_roi / 255.0
                    # 4. Reshape for the model (batch, height, width, channels)
                    input_data = normalized_roi.reshape(1, IMG_SIZE, IMG_SIZE, 3)

                    # --- Make Prediction ---
                    # prediction = model.predict(input_data, verbose=0)
                    kp_input = extract_123d_keypoints(hand_landmarks.landmark)
                    kp_input = kp_input.reshape(1, 123)
                    # 融合输入预测
                    prediction = model.predict([input_data, kp_input], verbose=0)

                    predicted_index = np.argmax(prediction[0])
                    confidence = np.max(prediction[0]) * 100
                    # predicted_letter = index_to_letter.get(predicted_index, '?')
                    last_predictions.append(predicted_index)
                    voted_index = max(set(last_predictions), key=last_predictions.count)
                    predicted_letter = index_to_letter.get(voted_index, '?')

                    # modify recognized_text
                    now = time.time()
                    if now - last_append_time >= 1:
                        if predicted_letter == 'del':
                            recognized_text = recognized_text[:-1]
                            last_append_time = now

                        elif predicted_letter == 'space':
                            recognized_text += ' '
                            last_append_time = now

                        elif predicted_letter.isalpha():
                            if len(recognized_text) == 0:
                                recognized_text += predicted_letter.upper()
                            else:
                                recognized_text += predicted_letter.lower()
                            last_append_time = now

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
    
    # --- Display the accumulated recognized text ---
    text_window = render_text_window(recognized_text)
    cv2.imshow("Recognized Text", text_window)

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
