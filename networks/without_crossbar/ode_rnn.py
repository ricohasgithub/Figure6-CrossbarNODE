
import torch
import torch.nn as nn
import torch.optim as optim

import matplotlib.pyplot as plt

from torchdiffeq import odeint

from utils.linear import Linear
from crossbar.crossbar import crossbar

class ODE_Func(nn.Module):

    def __init__(self, hidden_layer_size, cb):
        super(ODE_Func, self).__init__()
        self.linear = nn.Linear(hidden_layer_size, hidden_layer_size)
        self.nonlinear = nn.Tanh()

    def forward(self, t, x):
        # x = torch.transpose(x, 0, 1)
        out = self.nonlinear(self.linear(x))
        return out

    def use_cb(self, state):
        self.linear.use_cb(state)

class ODE_RNN(nn.Module):

    def __init__(self, input_size, hidden_layer_size, output_size, device_params):
        
        super(ODE_RNN, self).__init__()

        # Initialize model instance crossbar
        self.cb = crossbar(device_params)

        # Initialize initial hidden state
        self.h_0 = torch.zeros(hidden_layer_size, 1)
        self.hidden_layer_size = hidden_layer_size

        # Model layers
        self.linear_in = nn.Linear(input_size, hidden_layer_size)
        self.linear_hidden = nn.Linear(hidden_layer_size, hidden_layer_size)
        self.decoder = nn.Linear(hidden_layer_size, output_size)

        self.ode_func = ODE_Func(hidden_layer_size, self.cb)
        self.nonlinear = nn.Tanh()

    def forward(self, t, x, method="dopri5", step_size=20):
        
        t = t.reshape(-1).float()
        h = torch.zeros(1, self.hidden_layer_size)
        h_i = torch.zeros(1, self.hidden_layer_size)

        # RNN iteration
        for i, x_i in enumerate(x):
            x_i = torch.transpose(x_i, 0, 1)
            if i > 0:
                h_i = odeint(self.ode_func, h, t[i-1 : i+1])[1]
            h = self.nonlinear(self.linear_in(x_i) + self.linear_hidden(h_i))

        out = self.decoder(h)
        return out

    def use_cb(self, state):
        self.linear_in.use_cb(state)
        self.ode_func.use_cb(state)
        self.linear_hidden.use_cb(state)
        self.decoder.use_cb(state)

def train(model, data_gen, epochs):

    examples = data_gen.train_data

    optimizer = optim.Adam(model.parameters(), lr=0.01)
    loss_function = nn.MSELoss()

    loss_history = []

    for epoch in range(epochs):

        epoch_loss = []

        for i, (example, label) in enumerate(examples):
            
            optimizer.zero_grad()
            prediction = model(example[1], example[0])

            loss = loss_function(prediction, label.transpose(0, 1))
            epoch_loss.append(loss)
            loss.backward()
            optimizer.step()
        
        loss_history.append(sum(epoch_loss) / len(examples))
        epoch_loss = []

        print('Epoch {:04d} | Total Loss {:.6f}'.format(epoch, loss_history[epoch]))

    # Test
    seq = data_gen.test_start[0][0]
    t = data_gen.test_start[0][1]
    num_predict = 30
    length = num_predict

    dt = torch.sum(t[1:] - t[0:-1]) / (len(t) - 1)
    output = []
    all_t = []

    #model.use_cb(True)
    
    with torch.no_grad():
        for i in range(length):
            prediction = model((t + dt), seq).transpose(0, 1).reshape(1, -1, 1)
            seq = torch.cat((seq[1:], prediction), axis=0)
            all_t.append(t[-1].unsqueeze(0) + dt.unsqueeze(0))
            t = torch.cat((t[1:], t[-1].unsqueeze(0) + dt.unsqueeze(0)), axis=0)
            output.append(prediction)

    output, times = torch.cat(output, axis=0), torch.cat(all_t, axis=0)

    ax = plt.axes(projection='3d')

    o1, o2, o3 = output[:, 0].squeeze(), output[:, 1].squeeze(), times.squeeze()
    # o1, o2, o3 = output[:, 0].squeeze(), output[:, 1].squeeze(), output[:, 2].squeeze()
    ax.plot3D(o1, o2, o3, 'blue')
    ax.scatter3D(o1, o2, o3, 'blue')
    
    d1, d2, d3 = data_gen.y[0, :].squeeze(), data_gen.y[1, :].squeeze(), data_gen.x.squeeze()
    # d1, d2, d3 = data_gen.y[0, :].squeeze(), data_gen.y[1, :].squeeze(), data_gen.y[2, :].squeeze()
    ax.plot(d1, d2, d3, 'gray')
    ax.scatter3D(d1, d2, d3, 'gray')

    plt.savefig('./output/ode_rnn.png', dpi=600, transparent=True)

    return ax
