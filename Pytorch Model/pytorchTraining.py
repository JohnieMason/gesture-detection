import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Hyperparameters
input_size = 2  # (x, y) coordinates
hidden_size = 256  # Increased hidden size
num_layers = 2
num_classes = 4
dropout = 0.5
learning_rate = 0.0001  # Reduced learning rate
num_epochs = 20

# Define the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Model definition
class GestureRecognitionModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout=0.5):
        super(GestureRecognitionModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # LSTM layer
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)

        # Fully connected layer
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # Set initial hidden and cell states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size)

        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        return out

# Create the model
model = GestureRecognitionModel(input_size, hidden_size, num_layers, num_classes, dropout).to(device)

# Assuming you have preprocessed your data into sequences and labels tensors
# Example tensors for demonstration
# sequences = torch.randn(100, 24, input_size)  # 100 sequences, each of length 24 with 2 features (x, y)
# labels = torch.randint(0, num_classes, (100,))  # 100 labels

# Assuming `sequences` and `labels` are your preprocessed data tensors
# Uncomment and replace with actual data loading if necessary
# sequences, labels = preprocessor.preprocess_data_from_file("coordinates_data.txt")

# For demonstration, let's create dummy data
sequences = torch.randn(100, 24, input_size)  # 100 sequences, each of length 24 with 2 features (x, y)
labels = torch.randint(0, num_classes, (100,))  # 100 labels

train_dataset = TensorDataset(sequences, labels)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.01)  # Added weight decay

# Training loop
for epoch in range(num_epochs):
    model.train()
    for sequences, labels in train_loader:
        sequences = sequences.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(sequences)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping
        optimizer.step()

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

print("Training complete!")