import numpy as np
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from model import GestureRecognitionModel

# Load the preprocessed sequences and labels from .npy files
sequences = np.load('sequences.npy')
labels = np.load('labels.npy')

# Split the data into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(sequences, labels, test_size=0.2, random_state=42)

# Convert the data to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

# Define the model, loss function, and optimizer
model = GestureRecognitionModel()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Print the model summary
print(model)

# Define training parameters
num_epochs = 10
batch_size = 32

# Training loop
for epoch in range(num_epochs):
    model.train()
    for i in range(0, len(X_train), batch_size):
        batch_x = X_train[i:i+batch_size]
        batch_y = y_train[i:i+batch_size]

        # Debugging: Print shapes of tensors
        print(f"Batch x shape: {batch_x.shape}")
        print(f"Batch y shape: {batch_y.shape}")

        # Forward pass
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

# Evaluate the model on the testing data
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for i in range(0, len(X_test), batch_size):
        batch_x = X_test[i:i+batch_size]
        batch_y = y_test[i:i+batch_size]
        outputs = model(batch_x)
        _, predicted = torch.max(outputs.data, 1)
        _, labels_max = torch.max(batch_y, 1)
        total += batch_y.size(0)
        correct += (predicted == labels_max).sum().item()

    print(f"Test Accuracy: {100 * correct / total:.2f}%")

# Save the trained model and its weights
torch.save(model.state_dict(), "trained_model.pth")