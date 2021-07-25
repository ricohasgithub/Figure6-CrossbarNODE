
import torch
import torch.nn as nn
import torch.optim as optim

import random
import matplotlib.pyplot as plt

from torchdiffeq import odeint

from utils.linear import Linear
from utils.rnn_cell import GRU_Cell
from crossbar.crossbar import crossbar

class ODE_Func(nn.Module):

    def __init__(self, hidden_layer_size, cb):
        super(ODE_Func, self).__init__()
        self.linear = Linear(hidden_layer_size, hidden_layer_size, cb)
        self.linear2 = Linear(hidden_layer_size, hidden_layer_size, cb)
        self.nonlinear = nn.Tanh()

    def forward(self, t, x):
        # x = torch.transpose(x, 0, 1)
        out = self.linear2(self.nonlinear(self.linear(x)))
        return out

    def remap(self):
        self.linear.remap()
        self.linear2.remap()

    def use_cb(self, state):
        self.linear.use_cb(state)
        self.linear2.use_cb(state)

class ODE_RNN(nn.Module):

    def __init__(self, input_size, hidden_layer_size, output_size, device_params, method, step_size):
        
        super(ODE_RNN, self).__init__()

        # Initialize model instance crossbar
        self.cb = crossbar(device_params)

        # Initialize initial hidden state
        self.h_0 = torch.zeros(hidden_layer_size, 1)
        self.hidden_layer_size = hidden_layer_size

        # Model layers
        self.linear_in = Linear(input_size, hidden_layer_size, self.cb)
        self.linear_hidden = Linear(hidden_layer_size, hidden_layer_size, self.cb)
        self.linear_hidden2 = Linear(hidden_layer_size, hidden_layer_size, self.cb)
        self.decoder = Linear(hidden_layer_size, output_size, self.cb)
        self.method = method
        self.step_size = step_size

        #self.rnn_cell = nn.GRUCell(input_size, hidden_layer_size)

        self.ode_func = ODE_Func(hidden_layer_size, self.cb)
        self.nonlinear = nn.Tanh()

    def forward(self, t, x):
        
        t = t.reshape(-1).float()
        h_i = torch.zeros(self.hidden_layer_size, 1)
        h_ip = torch.zeros(self.hidden_layer_size, 1)

        # RNN iteration
        for i, x_i in enumerate(x):
            if i > 0:
                if self.step_size != None:
                    h_ip = odeint(self.ode_func, h_i, t[i-1 : i+1], method=self.method, options=dict(step_size=self.step_size))[1]
                else:
                    h_ip = odeint(self.ode_func, h_i, t[i-1 : i+1], method=self.method)[1]
                h_i = self.nonlinear(self.linear_hidden2(self.nonlinear(self.linear_in(x_i) + self.linear_hidden(h_ip))))
            #h_i = self.rnn_cell(x_i, h_ip)

        out = self.decoder(h_i)
        return out

    def use_cb(self, state):
        self.linear_in.use_cb(state)
        self.ode_func.use_cb(state)
        self.linear_hidden.use_cb(state)
        self.decoder.use_cb(state)

    def remap(self):
        self.linear_in.remap()
        self.linear_hidden.remap()
        self.linear_hidden2.remap()
        self.decoder.remap()

def train(model, data_gen, epochs):

    # model.use_cb(True)

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
            epoch_loss.append(loss)
            loss.backward()
            optimizer.step()

            # model.remap()
        
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