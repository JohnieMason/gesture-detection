import torch
import torch.nn as nn
import tensorflow as tf
import numpy as np

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

keras_model = tf.keras.models.load_model('trained_keras_model.keras')
keras_model.load_weights('trained_keras_model.weights.h5')

pytorch_model = SimpleRNNModel()

def transfer_weights(keras_layer, pytorch_layer, rnn_layer=False):
    weights = keras_layer.get_weights()
    print(f"Transferring weights for layer: {keras_layer.name}")
    print(f"Number of weights: {len(weights)}")

    if rnn_layer:
        pytorch_layer.weight_ih_l0.data = torch.from_numpy(np.transpose(weights[0])).float()
        pytorch_layer.weight_hh_l0.data = torch.from_numpy(np.transpose(weights[1])).float()
        if len(weights) > 2:
            pytorch_layer.bias_ih_l0.data = torch.from_numpy(weights[2]).float()
        else:
            pytorch_layer.bias_ih_l0.data = torch.zeros(pytorch_layer.bias_ih_l0.data.size()).float()
        pytorch_layer.bias_hh_l0.data = torch.zeros(pytorch_layer.bias_hh_l0.data.size()).float()
    else:
        pytorch_layer.weight.data = torch.from_numpy(np.transpose(weights[0])).float()
        pytorch_layer.bias.data = torch.from_numpy(weights[1]).float()

transfer_weights(keras_model.layers[0], pytorch_model.rnn1, rnn_layer=True)
transfer_weights(keras_model.layers[1], pytorch_model.rnn2, rnn_layer=True)

transfer_weights(keras_model.layers[2], pytorch_model.fc)

torch.save(pytorch_model.state_dict(), 'pytorch_model.pth')

print("Model conversion complete. PyTorch model saved as 'pytorch_model.pth'")