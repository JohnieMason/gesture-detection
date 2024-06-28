import torch

class GesturePreprocessor:
    def __init__(self, frame_width, frame_height, max_len, padding_value=0):
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.max_len = max_len
        self.padding_value = padding_value
        self.label_to_int = {"rl": 0, "lr": 1, "du": 2, "ud": 3}  # Map labels to integers

    def normalize_coordinates(self, sequence):
        sequence[:, 0] = sequence[:, 0] / self.frame_width  # Normalize X-coordinate
        sequence[:, 1] = sequence[:, 1] / self.frame_height  # Normalize Y-coordinate
        return sequence

    def redistribute_values(self, sequence):
        length = len(sequence)
        if length == self.max_len:
            return sequence

        fraction = (length - 1) / (self.max_len - 1)
        new_sequence = [sequence[0].tolist()]  # Ensure the first element is a list
        a = fraction
        while a < length - 1:
            lower = int(a)
            upper = lower + 1
            # Extract individual elements for interpolation
            lower_x, lower_y, _ = sequence[lower]
            upper_x, upper_y, _ = sequence[upper]

            # Perform interpolation for X and Y coordinates
            interpolated_x = lower_x + (upper_x - lower_x) * (a - lower)
            interpolated_y = lower_y + (upper_y - lower_y) * (a - lower)

            new_val = [
                interpolated_x.item(),  # Convert tensor to scalar
                interpolated_y.item(),  # Convert tensor to scalar
                sequence[lower][2].item()  # Convert tensor to scalar
            ]
            new_sequence.append(new_val)
            a += fraction
        if len(new_sequence) < self.max_len:
            new_sequence.append(sequence[-1].tolist())  # Ensure the last element is a list
        return torch.tensor(new_sequence, dtype=torch.float)

    def pad_sequences(self, sequences):
        padded_sequences = []
        for seq in sequences:
            if len(seq) < self.max_len:
                pad_len = self.max_len - len(seq)
                padding = torch.full((pad_len, seq.shape[1]), self.padding_value)
                padded_seq = torch.vstack((padding, seq))
            else:
                padded_seq = seq[:self.max_len]
            padded_sequences.append(padded_seq)
        return torch.stack(padded_sequences)

    def preprocess_data_from_file(self, filename):
        sequences = []
        labels = []
        with open(filename, 'r') as f:
            for line in f:
                try:
                    parts = line.strip().rsplit('], ', 1)
                    sequence_str, label_str = parts[0] + ']', parts[1]
                    sequence = eval(sequence_str)  # Assuming your data is comma-separated lists
                    label = label_str.strip()

                    if len(sequence) < 5:
                        print(f"Skipping short sequence: {sequence}")
                        continue

                    arr = torch.tensor([[item[0], item[1], 0] for item in sequence])
                    arr = self.normalize_coordinates(arr)
                    if label != 'x':
                        sequences.append(arr)
                        labels.append(self.label_to_int[label])

                except (SyntaxError, IndexError) as e:
                    print(f"Error parsing line: {line}\nError: {e}")
                    continue

        if not sequences or not labels:
            raise ValueError("No valid data found. Please check the data format and content.")

        sequences = torch.stack([self.redistribute_values(seq) for seq in sequences])
        labels = torch.tensor(labels)

        return sequences, labels

    def save_preprocessed_data(self, sequences, labels, sequence_file, label_file):
        torch.save(sequences, sequence_file)
        torch.save(labels, label_file)
        print(f"Preprocessed data saved to {sequence_file} and {label_file}")

preprocessor = GesturePreprocessor(frame_width=640, frame_height=480, max_len=24)
sequences, labels = preprocessor.preprocess_data_from_file("coordinates_data.txt")
preprocessor.save_preprocessed_data(sequences, labels, "preprocessed_sequences.pt", "preprocessed_labels.pt")