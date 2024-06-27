import torch
import torch.nn as nn

class PyTorchModel(nn.Module):
    def __init__(self):
        super(PyTorchModel, self).__init__()
        self.rnn1 = nn.RNN(input_size=2, hidden_size=10, batch_first=True)
        self.rnn2 = nn.RNN(input_size=10, hidden_size=10, batch_first=False)
        self.dropout = nn.Dropout(0.2)
        self.fc1 = nn.Linear(10, 2)  # Adjusted to match the Keras Dense layer for 2 classes
        self.fc2 = nn.Linear(10, 4)  # Output size 4 for 4 gesture classes as per Keras model

    def forward(self, x):
        x, _ = self.rnn1(x)
        x, _ = self.rnn2(x)
        x = self.dropout(x)
        x1 = self.fc1(x[:, -1, :])  # Assuming you want output after the last timestep
        x2 = self.fc2(x[:, -1, :])  # Assuming you want output after the last timestep
        return x1, x2

# Create an instance of the PyTorch model
pytorch_model = PyTorchModel()