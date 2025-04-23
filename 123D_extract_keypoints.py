import os
import cv2
import numpy as np
import mediapipe as mp

source_root = 'dataset/train'
target_root = 'dataset/train_keypoints_123d'

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1)
os.makedirs(target_root, exist_ok=True)

# Define the hand skeleton connection edges (20), refer to the topology of MediaPipe Hand
edges = [
    (0, 1), (1, 2), (2, 3), (3, 4),       # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),       # Index Finger
    (0, 9), (9,10), (10,11), (11,12),     # Middle finger
    (0,13), (13,14), (14,15), (15,16),    # Ring finger
    (0,17), (17,18), (18,19), (19,20)     # Pinky
]

for label in os.listdir(source_root):
    img_dir = os.path.join(source_root, label)
    keypoint_dir = os.path.join(target_root, label)
    os.makedirs(keypoint_dir, exist_ok=True)

    for filename in os.listdir(img_dir):
        if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue

        img_path = os.path.join(img_dir, filename)
        image = cv2.imread(img_path)
        if image is None:
            continue

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image_rgb)

        if results.multi_hand_landmarks:
            hand = results.multi_hand_landmarks[0]
            lm_list = [[lm.x, lm.y, lm.z] for lm in hand.landmark]  # 21x3

            original = np.array(lm_list).flatten()  # 63 dimensions
            diffs = []
            for i, j in edges:
                vec = np.subtract(lm_list[j], lm_list[i])  # dx,dy,dz
                diffs.extend(vec)

            keypoint_123d = np.concatenate([original, diffs])  # Total 123 dimensions
        else:
            keypoint_123d = np.zeros(123, dtype=np.float32)

        npy_name = os.path.splitext(filename)[0] + '.npy'
        np.save(os.path.join(keypoint_dir, npy_name), keypoint_123d)

hands.close()
print("Extraction completed, a total of 123 features")
