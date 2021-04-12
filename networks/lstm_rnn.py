
import torch
import torch.nn as nn

from utils.linear import Linear
from crossbar.crossbar import crossbar

class LSTM_RNN(nn.Module):

    def __init__(self, input_layer_size, hidden_layer_size, output_layer_size, device_params):

        super(LSTM_RNN, self).__init__()

        self.input_layer_size = input_layer_size
        self.hidden_layer_size = hidden_layer_size
        self.output_layer_size = output_layer_size

        self.cb = crossbar(device_params)

        self.lstm = nn.LSTM(input_layer_size, hidden_layer_size, num_layers=1, batch_first=True)
        self.linear = Linear(hidden_layer_size, output_layer_size, self.cb)
    
    def forward(self, x):
        output, hidden = self.lstm(x)
        output = output[:, 1, :]
        output = self.linear(torch.relu(output))
        return output

    def use_cb(self, state):
        self.linear_in.use_cb(state)
        self.linear_hidden.use_cb(state)
        self.solve.use_cb(state)
