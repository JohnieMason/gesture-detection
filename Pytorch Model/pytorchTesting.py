import cv2
import numpy as np
import time
import torch
import torch.nn as nn
from torch.nn import functional as F
import mediapipe as mp

class PyTorchModel(nn.Module):
    def __init__(self):
        super(PyTorchModel, self).__init__()
        self.rnn1 = nn.RNN(input_size=2, hidden_size=10, batch_first=True)
        self.rnn2 = nn.RNN(input_size=10, hidden_size=10, batch_first=False)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(10, 4)

    def forward(self, x):
        x, _ = self.rnn1(x)
        x, _ = self.rnn2(x)
        x = self.dropout(x)
        x = self.fc(x[:, -1, :])
        return x

pytorch_model = PyTorchModel()
pytorch_model.load_state_dict(torch.load('pytorch_model.pth'))
pytorch_model.eval()

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)

cap = cv2.VideoCapture(0)
frame_width = 640
frame_height = 480
cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

arr = []
labels = {0: "du", 1: "lr", 2: "rl", 3: "ud"}
hand_detected = False
confidence_threshold = 0.7

def normalize_coordinates(arr, frame_width, frame_height):
    arr[:, 0] = arr[:, 0] / frame_width
    arr[:, 1] = arr[:, 1] / frame_height
    return arr

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

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Error: Unable to capture frame.")
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        landmark_points = [
            (int(l.x * frame.shape[1]), int(l.y * frame.shape[0]), time.time_ns() / 1e6)
            for l in hand_landmarks.landmark
        ]

        landmark_points_np = np.array(landmark_points, dtype=np.float32)
        brect = cv2.boundingRect(cv2.convexHull(landmark_points_np[:, :2].astype(np.int32)))
        cv2.rectangle(frame, (brect[0], brect[1]), (brect[0] + brect[2], brect[1] + brect[3]), (0, 255, 0), 2)

        center_x = brect[0] + brect[2] // 2
        center_y = brect[1] + brect[3] // 2
        cv2.putText(frame, f'Center: ({center_x}, {center_y})', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0),
                    2, cv2.LINE_AA)

        if not hand_detected:
            hand_detected = True
            print("Hand detected. Collecting data...")

        arr.append([center_x, center_y, time.time_ns() / 1e6])

    else:
        if hand_detected and len(arr) > 0:
            print("Hand no longer detected. Preprocessing and predicting...")

            data = np.array(arr)
            data = normalize_coordinates(data, frame_width, frame_height)
            data = redistribute_values(data, 24)
            data = data[:, :2]
            data = torch.tensor(data, dtype=torch.float32).unsqueeze(0)

            with torch.no_grad():
                outputs = pytorch_model(data)
                _, predicted = torch.max(outputs.data, 1)

            gesture_label = labels[predicted.item()]

            print("Prediction:", gesture_label)

            arr.clear()
            hand_detected = False

    cv2.imshow('Hand Tracking', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()