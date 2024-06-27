import cv2
import mediapipe as mp
import numpy as np
import time

# Access the mediapipe hands module
mp_hands = mp.solutions.hands
# Initialize the hands module with certain parameters
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.3)

# Initialize a video capture from the default webcam (0 selects the default)
cap = cv2.VideoCapture(0)
# The frame dimensions are 640 x 480
frame_width = 640
frame_height = 480
cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

# arr stores the array of coordinates of the center of the bounding box
arr = []
frame_count = 0

# Open a file coordinates_data.txt for writing hand coordinates
with open('coordinates_data.txt', 'w') as f:
    # Continuously capture frames from the webcam
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Error: Unable to capture frame.")
            break
        # Convert each frame from BGR to RGB format
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = hands.process(frame_rgb)

        # This block runs only if hand landmarks have been detected in the current frame
        if results.multi_hand_landmarks:
            # frame_count keeps track of the number of frames processed since it was last reset.
            # Increment frame_count by 1 every time a frame is processed.
            frame_count += 1
            # Reset frame_count to 0 every time it reaches 5
            # This ensures that the line `arr.append([center_x, center_y, frame_time])` runs once every 5 frames
            frame_count %= 5

            # Retrieve the landmarks of the first detected hand.
            hand_landmarks = results.multi_hand_landmarks[0]

            # convert the normalized coordinates of hand landmarks into pixel coordinates
            # that are suitable for drawing on the image
            landmark_points = [(int(l.x * frame.shape[1]), int(l.y * frame.shape[0])) for l in hand_landmarks.landmark]

            # Convert the list landmark_points into a NumPy array.
            # dtype=np.int32 ensures that the array elements are 32-bit integers
            landmark_points_np = np.array(landmark_points, dtype=np.int32)

            # Draw bounding box around hand (rectangle)
            brect = cv2.boundingRect(cv2.convexHull(landmark_points_np))
            cv2.rectangle(frame, (brect[0], brect[1]), (brect[0]+brect[2], brect[1]+brect[3]), (0, 255, 0), 2)

            # Calculate center of bounding box
            center_x = brect[0] + brect[2] // 2
            center_y = brect[1] + brect[3] // 2
            # Get the current time in milliseconds.
            frame_time = time.time_ns() / 1e6

            # The coordinates and frame time are appended only once every 5 frames
            if frame_count == 0:
                arr.append([center_x, center_y, frame_time])
            # Display the center coordinates on the camera feed
            cv2.putText(frame, f'Center: ({center_x}, {center_y})', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # If no hand landmarks have been detected in the current frame and arr has stored data,
        # write the data to the file and clear arr.
        else:
            if len(arr) > 0:
                f.write(f"{arr}\n")
                print(len(arr))
                arr.clear()
        cv2.imshow('Hand Tracking', frame)

        # If 'q' is pressed, exit the loop (close the camera window)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release the video capture and close all OpenCV windows.
cap.release()
cv2.destroyAllWindows()