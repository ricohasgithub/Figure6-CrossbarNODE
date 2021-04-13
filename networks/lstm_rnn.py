
import torch
import torch.nn as nn
import torch.optim as optim

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
        self.linear = nn.Linear(hidden_layer_size, output_layer_size)
    
    def forward(self, x):
        out, h = self.lstm(x)
        # out = out[:, 1, :]
        out = self.linear(torch.relu(out))
        return out

    def use_cb(self, state):
        self.linear_in.use_cb(state)
        self.linear_hidden.use_cb(state)
        self.solve.use_cb(state)

def train(model, data_gen, iters):

    optimizer = optim.Adam(model.parameters(), lr=0.01)
    loss_func = nn.MSELoss()

    for itr in range(1, iters + 1):

        optimizer.zero_grad()
        batch_y0, batch_t, batch_y = data_gen.get_random_batch()

        print(batch_y0.size())

        pred_y = model(batch_y)

        loss = loss_func(pred_y, batch_y)
        loss.backward()
        optimizer.step()

        print('Iter {:04d} | Total Loss {:.6f}'.format(itr, loss))

    with torch.no_grad():
        pred_y = model(torch.tensor(data_gen.true_y0))
        data_gen.plot_prediction(data_gen.true_y, pred_y)