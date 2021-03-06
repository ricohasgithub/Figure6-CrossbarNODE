
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

class ODE_RNN(nn.Module):

    def __init__(self, input_size, hidden_layer_size, output_size, device_params, method, step_size):
        
        super(ODE_RNN, self).__init__()

        # Initialize model instance crossbar
        self.cb = crossbar(device_params)

        # Initialize initial hidden state
        self.h_0 = torch.zeros(hidden_layer_size, 1)
        self.hidden_layer_size = hidden_layer_size
        self.output_size = output_size
        self.method = method
        self.step_size = step_size

        # Model layers
        self.linear_in = Linear(input_size, hidden_layer_size, self.cb)
        self.linear_hidden = Linear(hidden_layer_size, hidden_layer_size, self.cb)
        self.decoder = Linear(hidden_layer_size, output_size, self.cb)

        self.rnn_cell = GRU_Cell(input_size, hidden_layer_size, self.cb)

        self.ode_func = ODE_Func(hidden_layer_size, self.cb)
        self.nonlinear = nn.Tanh()

    def forward(self, t, x):
        
        t = t.reshape(-1).float()
        N = t.shape[0]

        # Model current output hidden state dynamics
        h_i = torch.zeros(self.hidden_layer_size, 1)
        h_ip = torch.zeros(self.hidden_layer_size, 1)
        output = torch.zeros(N, self.output_size)

        # Initial layer (h0)
        if t[0] > 0:

            h_ip = odeint(self.ode_func, h_i, torch.tensor([0.0, t[0]]), method=self.method)[1]

            out = self.linear_hidden(h_ip)
            out = self.decoder(out)
            out = self.nonlinear(out)
            output[0] = out.reshape(-1)

            h_i = self.rnn_cell(x[0], h_ip)
            # h_i = self.rnn_cell(x[0].transpose(0, 1), h_ip.transpose(0, 1))
            h_i = torch.transpose(h_i, 0, 1)

        # RNN iteration
        for i in range(1, N):

            x_i = x[i]
            h_ip = odeint(self.ode_func, h_i, t[i-1 : i+1], method=self.method)[1]

            out = self.linear_hidden(h_ip)
            out = self.decoder(out)
            out = self.nonlinear(out)
            output[i] = out.reshape(-1)

            h_i = self.rnn_cell(x_i, h_ip)
            # h_i = self.rnn_cell(x_i.transpose(0, 1), h_ip.transpose(0, 1))
            h_i = torch.transpose(h_i, 0, 1)

        return output, h_i

    def remap(self):
        self.linear_in.remap()
        self.ode_func.remap()
        self.linear_hidden.remap()

    def use_cb(self, state):
        self.linear_in.use_cb(state)
        self.ode_func.use_cb(state)
        self.linear_hidden.use_cb(state)
        self.decoder.use_cb(state)

def train(model, data_gen, epochs):

    #model.use_cb(True)

    examples = data_gen.train_data

    optimizer = optim.Adam(model.parameters(), lr=0.015)
    loss_function = nn.MSELoss()

    loss_history = []

    for epoch in range(epochs):

        random.shuffle(examples)
        epoch_loss = []

        for i, (example, label) in enumerate(examples):
            
            optimizer.zero_grad()
            prediction = model(example[1], example[0])[0].transpose(0, 1)

            loss = loss_function(prediction, label)
            epoch_loss.append(loss)
            loss.backward()
            optimizer.step()
        
        loss_history.append(sum(epoch_loss) / len(examples))
        epoch_loss = []

        print('Epoch {:04d} | Total Loss {:.6f}'.format(epoch, loss_history[epoch]))

    # Test
    seq = data_gen.test_data[0][0][0]
    print("seq: ", seq.size())
    print(seq)
    times = data_gen.test_data[0][0][1]
    # num_predict currently not being used
    num_predict = 50
    length = num_predict

    output = []

    # TODO: make seq same length as times (30)
    with torch.no_grad():
        prediction = model(times, seq)[0]
        print("pred2: ", prediction.size())
        print(prediction)
        output.append(prediction)
    
    output = torch.cat(output, axis=0)

    # ax = plt.axes(projection='3d')

    o1, o2, o3 = output[:, 0].squeeze(), output[:, 1].squeeze(), times.squeeze()
    # o1, o2, o3 = output[:, 0].squeeze(), output[:, 1].squeeze(), output[:, 2].squeeze()
    # ax.plot3D(o1, o2, o3, 'red')
    # ax.scatter3D(o1, o2, o3, 'red')
    
    # d1, d2, d3 = data_gen.y[0, :].squeeze(), data_gen.y[1, :].squeeze(), data_gen.x.squeeze()
    # # d1, d2, d3 = data_gen.y[0, :].squeeze(), data_gen.y[1, :].squeeze(), data_gen.y[2, :].squeeze()
    # # ax.plot3D(d1, d2, d3, 'gray')
    # ax.plot3D(data_gen.true_x, data_gen.true_y, data_gen.true_z, 'gray')
    # ax.scatter3D(d1, d2, d3, 'gray')

    # plt.savefig('./output/ode_rnn.png', dpi=600, transparent=True)

    # return loss_history, ax
    return loss_history, [o1, o2, o3]
