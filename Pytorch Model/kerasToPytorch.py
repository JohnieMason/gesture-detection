import torch
import torch.nn as nn
import h5py  # for loading Keras weights

class SimpleRNNModel(nn.Module):
    def __init__(self):
        super(SimpleRNNModel, self).__init__()
        self.rnn1 = nn.RNN(input_size=2, hidden_size=10, batch_first=True)
        self.rnn2 = nn.RNN(input_size=10, hidden_size=10, batch_first=False)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(10, 4)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        out, _ = self.rnn1(x)
        # Access the last element of the sequence using slicing
        out = out[:, -1, :]  # Select last element from each sequence
        out, _ = self.rnn2(out.unsqueeze(1))
        out = self.dropout(out)
        out = self.fc(out.squeeze(1))
        out = self.softmax(out)
        return out

# Load Keras model weights (assuming the model is saved as 'trained_keras_model.h5')
with h5py.File('trained_keras_model.weights.h5', 'r') as f:
    keras_weights = f.get('layer_names')

# Define the PyTorch model
pytorch_model = SimpleRNNModel()

def transfer_weights(keras_layer, pytorch_layer, rnn_layer=False):
    weights = h5py.File('trained_keras_model.weights.h5', 'r')[keras_layer.name]['kernel:0']  # Access weights from h5 file
    weights = np.array(weights)
    print(f"Transferring weights for layer: {keras_layer.name}")
    print(f"Number of weights: {len(weights)}")

    if rnn_layer:
        pytorch_layer.weight_ih_l0.data = torch.from_numpy(weights).float()  # No transpose for weight_ih
        pytorch_layer.weight_hh_l0.data = torch.from_numpy(np.transpose(weights[1:])).float()  # Transpose for weight_hh
        if len(weights) > 2:
            pytorch_layer.bias_ih_l0.data = torch.from_numpy(weights[0]).float()  # Access bias from weights[0]
        else:
            pytorch_layer.bias_ih_l0.data = torch.zeros(pytorch_layer.bias_ih_l0.data.size()).float()
        pytorch_layer.bias_hh_l0.data = torch.zeros(pytorch_layer.bias_hh_l0.data.size()).float()
    else:
        pytorch_layer.weight.data = torch.from_numpy(np.transpose(weights)).float()
        pytorch_layer.bias.data = torch.from_numpy(weights[-1]).float()  # Access bias from last element

print('keras_weights', keras_weights)
# Transfer weights from Keras model to PyTorch model
for i, layer in enumerate(pytorch_model.modules()):
    # Check if layer is a weights layer (excluding softmax)
    if hasattr(layer, 'weight'):
        if layer.weight.requires_grad:  # Only transfer weights for trainable layers
            rnn_layer = isinstance(layer, nn.RNN)
            print(keras_weights[i])
            transfer_weights(keras_weights[i], layer, rnn_layer)

torch.save(pytorch_model.state_dict(), 'pytorch_model.pth')

print("Model conversion complete. PyTorch model saved as 'pytorch_model.pth'")