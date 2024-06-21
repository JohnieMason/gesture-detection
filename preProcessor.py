import numpy as np
import json
import tensorflow as tf

# scale the coordinates to be between 0 and 1 by dividing by the frame width and height.
def normalize_coordinates(arr, frame_width, frame_height):
    arr[:, 0] = arr[:, 0] / frame_width
    arr[:, 1] = arr[:, 1] / frame_height
    return arr

# redistribute the values of each sequence to a specified number of points
# (24 in our case) using linear interpolation
def redistribute_values(sequence:list[list|np.ndarray], points:int) -> np.ndarray:
    length = len(sequence)
    if points == length:
        return sequence
    fraction = (length - 1)/(points - 1)
    new_sequence = [sequence[0]]
    a = fraction
    while a < length - 1:
        lower = int(a)
        upper = lower + 1
        new_val = [(sequence[lower][0] + (sequence[upper][0] - sequence[lower][0]) * (a - lower)), (sequence[lower][1] + (sequence[upper][1] - sequence[lower][1]) * (a - lower))]
        new_sequence.append(new_val)
        a += fraction
    if len(new_sequence) < points:
        new_sequence.append(sequence[-1])
    return np.array(new_sequence)

# pad the sequences to a specified maximum length with a padding value of 0
# sequences must be prepaded to match the LSTM layer requirement
def pad_sequences(sequences, max_len, padding_value=0):
    # Since we are using LSTM layer as the input, we must prepad rather than post pad
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
    # open the file coordinates_data.txt
    with open(file_path, 'r') as f:
        # for each line in the file
        for line in f:
            try:
                # Split the line into sequence and label parts
                parts = line.strip().rsplit('], ', 1)
                sequence_str, label_str = parts[0] + ']', parts[1]
                # Convert the sequence part from JSON string to list
                sequence = json.loads(sequence_str)
                # Clean the label string
                label = label_str.strip()

                # Skip sequences shorter than the minimum length
                if len(sequence) < min_len:
                    print(f"Skipping short sequence: {sequence}")
                    continue

                # Convert the sequence to a NumPy array
                arr = np.array([[item[0], item[1]] for item in sequence], dtype=np.float32)
                # Normalize the coordinates
                arr = normalize_coordinates(arr, frame_width, frame_height)
                # Only include sequences with valid labels
                if label != 'x':
                    sequences.append(arr)
                    labels.append(label)

            except (json.JSONDecodeError, IndexError) as e:
                print(f"Error parsing line: {line}\nError: {e}")
                continue

    # Ensure there is valid data
    if len(sequences) == 0 or len(labels) == 0:
        raise ValueError("No valid data found in the file. Please check the data format and content.")

    # Redistribute sequences to the desired length
    temp = [redistribute_values(sequences[i], max_len) for i in range(len(sequences))]
    sequences = np.array(temp)
    # Get unique labels and sort them
    unique_labels = sorted(set(labels))
    # Map labels to integers
    label_to_int = {label: i for i, label in enumerate(unique_labels)}
    labels = np.array([label_to_int[label] for label in labels])

    # Number of gesture classes
    num_classes = 2
    # Convert labels to one-hot encoding
    labels = tf.keras.utils.to_categorical(labels, num_classes=num_classes)

    return sequences, labels

try:
    # Load and preprocess the data
    sequences, labels = load_and_preprocess_data('coordinates_data.txt')
    print("Preprocessing complete. Data shape:", sequences.shape, labels.shape)
    # Save the sequences and labels to .npy files
    np.save('sequences.npy', sequences)
    np.save('labels.npy', labels)

    print("Data saved to 'sequences.npy' and 'labels.npy'.")

# Print any errors that occur during preprocessing
except Exception as e:
    print("Error during preprocessing:", str(e))
