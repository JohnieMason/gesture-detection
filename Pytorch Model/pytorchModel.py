import torch
import torch.nn as nn

class SimpleRNNModel(nn.Module):
    def __init__(self):
        super(SimpleRNNModel, self).__init__()
        self.rnn1 = nn.RNN(input_size=2, hidden_size=10, batch_first=True)
        self.rnn2 = nn.RNN(input_size=10, hidden_size=10, batch_first=False)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(10, 4)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x, _ = self.rnn1(x)
        x, _ = self.rnn2(x[:, -1, :].unsqueeze(1))
        x = x.squeeze(1)
        x = self.dropout(x)
        x = self.fc(x)
        x = self.softmax(x)
        return x

pytorch_model = SimpleRNNModel()