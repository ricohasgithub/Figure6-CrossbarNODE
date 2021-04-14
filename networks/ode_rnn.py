
import torch
import torch.nn as nn
import torch.optim as optim

import matplotlib as plt

from torchdiffeq import odeint

from utils.linear import Linear
from crossbar.crossbar import crossbar

class ODE_Func(nn.Module):

    def __init__(self, hidden_layer_size, cb):
        super(ODE_Func, self).__init__()
        self.linear = Linear(hidden_layer_size, hidden_layer_size, cb)
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
        self.linear_in = Linear(input_size, hidden_layer_size, self.cb)
        self.linear_hidden = Linear(hidden_layer_size, hidden_layer_size, self.cb)
        self.decoder = Linear(hidden_layer_size, output_size, self.cb)

        self.ode_func = ODE_Func(hidden_layer_size, self.cb)
        self.nonlinear = nn.Tanh()

    def forward(self, t, x, method="dopri5", step_size=20):
        
        t = t.reshape(-1).float()
        h = torch.zeros(self.hidden_layer_size, 1)
        h_i = torch.zeros(self.hidden_layer_size, 1)

        # RNN iteration
        for i, x_i in enumerate(x):
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

    #model.use_cb(True)

    examples = data_gen.train_data

    optimizer = optim.Adam(model.parameters(), lr=0.01)
    loss_function = nn.MSELoss()

    loss_history = []

    for epoch in range(epochs):

        epoch_loss = []

        for i, (example, label) in enumerate(examples):
            
            optimizer.zero_grad()
            prediction = model(example[1], example[0])

            loss = loss_function(prediction, label)
            epoch_loss.append(loss)
            loss.backward()
            optimizer.step()
        
        loss_history.append(sum(epoch_loss) / len(examples))
        epoch_loss = []

        print('Epoch {:04d} | Total Loss {:.6f}'.format(epoch, loss_history[epoch]))

    # Test
    seq = data_gen.test_start[0][0]
    t = data_gen.test_start[0][1]
    length = 30

    dt = torch.sum(t[1:] - t[0:-1]) / (len(t) - 1)
    output = []
    all_t = []

    # model.use_cb(True)
    
    with torch.no_grad():
        for i in range(length):
            prediction = model((t + dt), seq).reshape(1, -1, 1)
            seq = torch.cat((seq[1:], prediction), axis=0)
            all_t.append(t[-1].unsqueeze(0) + dt.unsqueeze(0))
            t = torch.cat((t[1:], t[-1].unsqueeze(0) + dt.unsqueeze(0)), axis=0)
            output.append(prediction)

    output, times = torch.cat(output, axis=0), torch.cat(all_t, axis=0)

    ax = plt.axes(projection='3d')

    o1, o2, o3 = output[:, 0].squeeze(), output[:, 1].squeeze(), times.squeeze()
    ax.plot3D(o1, o2, o3, 'gray')
    
    d1, d2, d3 = data_gen.y[0, :].squeeze(), data_gen.y[1, :].squeeze(), data_gen.x.squeeze()
    ax.plot3D(d1, d2, d3, 'orange')

    plt.show()
