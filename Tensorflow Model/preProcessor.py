import numpy as np
import json
import tensorflow as tf

# Scale the coordinates to be between 0 and 1 by dividing by the frame width and height
def normalize_coordinates(arr, frame_width, frame_height):
    arr[:, 0] = arr[:, 0] / frame_width
    arr[:, 1] = arr[:, 1] / frame_height
    return arr

# Redistribute the values of each sequence to a specified number of points using linear interpolation
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

# Pad the sequences to a specified maximum length with a padding value of 0
def pad_sequences(sequences, max_len, padding_value=0):
    padded_sequences = []
    for seq in sequences:
        if len(seq) < max_len:
            pad_len = max_len - len(seq)
            padding = np.full((pad_len, seq.shape[1]), padding_value)
            padded_seq = np.vstack((padding, seq))
        else:
            padded_seq = seq[:max_len]
        padded_sequences.append(padded_seq)
    return np.array(padded_sequences)

def load_and_preprocess_data(file_path, frame_width=640, frame_height=480, max_len=24, min_len=5):
    sequences = []
    labels = []
    with open(file_path, 'r') as f:
        for line in f:
            try:
                parts = line.strip().rsplit('], ', 1)
                sequence_str, label_str = parts[0] + ']', parts[1]
                sequence = json.loads(sequence_str)
                label = label_str.strip()

                if len(sequence) < min_len:
                    print(f"Skipping short sequence: {sequence}")
                    continue

                arr = np.array([[item[0], item[1], 0] for item in sequence], dtype=np.float32)
                arr = normalize_coordinates(arr, frame_width, frame_height)
                if label != 'x':
                    sequences.append(arr)
                    labels.append(label)

            except (json.JSONDecodeError, IndexError) as e:
                print(f"Error parsing line: {line}\nError: {e}")
                continue

    if len(sequences) == 0 or len(labels) == 0:
        raise ValueError("No valid data found in the file. Please check the data format and content.")

    temp = [redistribute_values(sequences[i], max_len) for i in range(len(sequences))]
    sequences = np.array(temp)

    unique_labels = sorted(set(labels))
    label_to_int = {label: i for i, label in enumerate(["rl", "lr", "du", "ud"])}
    labels = np.array([label_to_int[label] for label in labels])

    num_classes = 4
    labels = tf.keras.utils.to_categorical(labels, num_classes=num_classes)

    return sequences, labels

