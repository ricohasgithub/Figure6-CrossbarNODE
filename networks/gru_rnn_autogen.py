
import torch
import torch.nn as nn
import torch.optim as optim

import random
import matplotlib.pyplot as plt

from torchdiffeq import odeint

from utils.linear import Linear
from utils.rnn_cell import GRU_Cell
from crossbar.crossbar import crossbar

class GRU_RNN(nn.Module):

    def __init__(self, input_size, hidden_layer_size, output_size, device_params):

        super(GRU_RNN, self).__init__()

        self.input_size = input_size
        self.hidden_layer_size = hidden_layer_size
        self.output_size = output_size

        self.cb = crossbar(device_params)

        self.rnn_cell = GRU_Cell(input_size, hidden_layer_size, self.cb)
        self.linear = Linear(hidden_layer_size, output_size, self.cb)
    
    def forward(self, t, x, method="dopri5", step_size=20):

        t = t.reshape(-1).float()
        N = t.shape[0]

        # Model current output hidden state dynamics
        h_i = torch.zeros(self.hidden_layer_size, 1)

        # Initial layer (h0)
        if t[0] > 0:

            h_i = self.rnn_cell(x[0], h_i)
            h_i = torch.transpose(h_i, 0, 1)
            out = self.linear(torch.relu(h_i))

        for i in range(1, N):

            h_i = self.rnn_cell(x[0], h_i)
            h_i = torch.transpose(h_i, 0, 1)
            out = self.linear(torch.relu(h_i))

        return out

    def use_cb(self, state):
        self.linear_in.use_cb(state)
        self.linear_hidden.use_cb(state)
        self.solve.use_cb(state)

    def remap(self):
        self.linear_in.remap()
        self.linear_hidden.remap()
        self.solve.remap()

def train(model, data_gen, epochs):

    #model.use_cb(True)

    examples = data_gen.train_data

    optimizer = optim.Adam(model.parameters(), lr=0.01)
    loss_function = nn.MSELoss()

    loss_history = []

    for epoch in range(epochs):

        random.shuffle(examples)
        epoch_loss = []

        for i, (example, label) in enumerate(examples):
            
            optimizer.zero_grad()
            prediction = model(example[1], example[0])

            loss = loss_function(prediction, label)
            loss.backward()
            optimizer.step()

            epoch_loss.append(loss.detach().numpy())

            #model.remap()
        
        loss_history.append(sum(epoch_loss) / len(examples))
        epoch_loss = []

        print('Epoch {:04d} | Total Loss {:.6f}'.format(epoch, loss_history[epoch]))

    return loss_history


def test(model, data_gen):
    
    # model.use_cb(True)

    # Test
    seq = data_gen.test_start
    t = data_gen.test_start[0][0][1]
    num_predict = 30
    length = num_predict

    dt = torch.sum(t[1:] - t[0:-1]) / (len(t) - 1)
    output = []
    all_t = []
    
    with torch.no_grad():
        for i, (example, label) in enumerate(seq):
            prediction = model(example[1], example[0]).reshape(1, -1, 1)
            output.append(prediction)
            all_t.append(example[1].unsqueeze(0)[0][9])

    output, times = torch.cat(output, axis=0), torch.cat(all_t, axis=0)

    print(output.size())
    print(times.size())

    o1, o2, o3 = output[:, 0].squeeze(), output[:, 1].squeeze(), times.squeeze()
    return [o1, o2, o3]