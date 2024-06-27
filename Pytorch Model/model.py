import torch
import torch.nn as nn

class GestureRecognitionModel(nn.Module):
    def __init__(self):
        super(GestureRecognitionModel, self).__init__()
        self.lstm = nn.LSTM(input_size=3, hidden_size=64, batch_first=True)
        self.dense1 = nn.Linear(64, 64)
        self.dense2 = nn.Linear(64, 4)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x, _ = self.lstm(x)
        x = x[:, -1, :]
        x = self.relu(self.dense1(x))
        x = self.softmax(self.dense2(x))
        return x

model = GestureRecognitionModel()
print(model)