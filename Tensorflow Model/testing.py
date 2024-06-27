import cv2
import mediapipe as mp
import numpy as np
import time
from tensorflow.keras.models import load_model
import json

# Load the pre-trained Keras model for gesture recognition
model = load_model("trained_model.keras")

# Initialize MediaPipe hands module for hand tracking
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)

# Initialize video capture from default camera (index 0)
cap = cv2.VideoCapture(0)

# Set frame dimensions for video capture
frame_width = 640
frame_height = 480
cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

# Initialize an empty list to store tracked hand gesture data
arr = []

# Initialize frame count to control the frequency of data collection
frame_count = 0

# Define gesture labels corresponding to model output
labels = {0: "du", 1: "lr", 2: "rl", 3: "ud"}

# Flag to track if hand is detected
hand_detected = False

# Confidence threshold for valid gesture
confidence_threshold = 0.7

# Function to normalize landmark coordinates to frame dimensions
def normalize_coordinates(arr, frame_width, frame_height):
    arr[:, 0] = arr[:, 0] / frame_width
    arr[:, 1] = arr[:, 1] / frame_height
    return arr

# Function to redistribute sequence to a fixed number of points using linear interpolation
def redistribute_values(sequence, points):
    length = len(sequence)
    if points == length:
        return sequence
    fraction = (length - 1) / (points - 1)
    new_sequence = [sequence[0]]
    a = fraction
    while a < length - 1:
        lower = int(a)
        upper = lower + 1
        new_val = [
            (sequence[lower][0] + (sequence[upper][0] - sequence[lower][0]) * (a - lower)),
            (sequence[lower][1] + (sequence[upper][1] - sequence[lower][1]) * (a - lower)),
            sequence[lower][2] + (sequence[upper][2] - sequence[lower][2]) * (a - lower)
        ]
        new_sequence.append(new_val)
        a += fraction
    if len(new_sequence) < points:
        new_sequence.append(sequence[-1])
    return np.array(new_sequence)

# Main loop to capture and process video frames
while cap.isOpened():
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Check if frame is successfully captured
    if not ret:
        print("Error: Unable to capture frame.")
        break

    # Convert BGR frame to RGB for MediaPipe
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame to detect hand landmarks
    results = hands.process(frame_rgb)

    # Check if hand landmarks are detected
    if results.multi_hand_landmarks:
        # Get landmarks of the first detected hand
        hand_landmarks = results.multi_hand_landmarks[0]

        # Extract landmark coordinates and add timestamp
        landmark_points = [
            (int(l.x * frame.shape[1]), int(l.y * frame.shape[0]), time.time_ns() / 1e6)
            for l in hand_landmarks.landmark
        ]

        # Convert landmarks to numpy array for further processing
        landmark_points_np = np.array(landmark_points, dtype=np.float32)

        # Draw bounding box around the hand using convex hull of landmark points
        brect = cv2.boundingRect(cv2.convexHull(landmark_points_np[:, :2].astype(np.int32)))
        cv2.rectangle(frame, (brect[0], brect[1]), (brect[0] + brect[2], brect[1] + brect[3]), (0, 255, 0), 2)

        # Calculate center coordinates of the bounding box
        center_x = brect[0] + brect[2] // 2
        center_y = brect[1] + brect[3] // 2

        # Display center coordinates on the frame
        cv2.putText(frame, f'Center: ({center_x}, {center_y})', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0),
                    2, cv2.LINE_AA)

        # If hand is detected, start collecting data
        if not hand_detected:
            hand_detected = True
            print("Hand detected. Collecting data...")

        # Store center coordinates and timestamp
        arr.append([center_x, center_y, time.time_ns() / 1e6])

    else:
        # If hand was detected and is now not detected, process and predict gesture
        if hand_detected and len(arr) > 0:
            print("Hand no longer detected. Preprocessing and predicting...")

            # Normalize and preprocess data
            data = np.array(arr)
            data = normalize_coordinates(data, frame_width, frame_height)
            data = redistribute_values(data, 24)
            data = data[:, :2]  # Use only (x, y) coordinates
            data = np.expand_dims(data, axis=0)

            # Predict gesture using the loaded model
            prediction = model.predict(data)

            # Check confidence level
            confidence = np.max(prediction)
            if confidence >= confidence_threshold:
                gesture_label = labels[np.argmax(prediction)]
            else:
                gesture_label = "not a valid gesture"

            print("Prediction:", gesture_label, " Confidence:", confidence)

            # Reset data collection
            arr.clear()
            hand_detected = False

    # Display the annotated frame with detected landmarks and predicted gesture
    cv2.imshow('Hand Tracking', frame)

    # Exit the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()