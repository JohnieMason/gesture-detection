import torch
import torch.nn as nn
import numpy as np
from sklearn.model_selection import train_test_split

# Load preprocessed data
sequences = np.load('sequences.npy')  # Load preprocessed sequences
labels = np.load('labels.npy')  # Load preprocessed labels (one-hot encoded)

# Convert to PyTorch tensors
sequences = torch.tensor(sequences, dtype=torch.float32)
labels = torch.tensor(np.argmax(labels, axis=1), dtype=torch.long)  # Convert one-hot labels to integer labels

# Split the data into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(sequences, labels, test_size=0.2, random_state=42)

# Define your PyTorch model class
class PyTorchModel(nn.Module):
    def __init__(self):
        super(PyTorchModel, self).__init__()
        self.rnn1 = nn.RNN(input_size=2, hidden_size=10, batch_first=True)
        self.rnn2 = nn.RNN(input_size=10, hidden_size=10, batch_first=False)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(10, 4)  # Output size 4 for 4 gesture classes

    def forward(self, x):
        x, _ = self.rnn1(x)
        x, _ = self.rnn2(x)
        x = self.dropout(x)
        x = self.fc(x[:, -1, :])  # Output after the last timestep
        return x

# Instantiate the model
pytorch_model = PyTorchModel()

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(pytorch_model.parameters(), lr=0.001)

# Training loop
epochs = 100
for epoch in range(epochs):
    pytorch_model.train()
    optimizer.zero_grad()

    # Forward pass
    outputs = pytorch_model(X_train)

    # Calculate loss
    loss = criterion(outputs, y_train)

    # Backward pass and optimize
    loss.backward()
    optimizer.step()

    # Print progress (optional)
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')

# Evaluate the model on the testing data
pytorch_model.eval()  # Switch to evaluation mode
with torch.no_grad():
    test_outputs = pytorch_model(X_test)
    test_loss = criterion(test_outputs, y_test)
    _, predicted = torch.max(test_outputs, 1)
    accuracy = (predicted == y_test).sum().item() / y_test.size(0)

print(f'Test Loss: {test_loss.item():.4f}, Test Accuracy: {accuracy:.4f}')

# Save the trained model
torch.save(pytorch_model.state_dict(), 'pytorch_model_trained.pth')