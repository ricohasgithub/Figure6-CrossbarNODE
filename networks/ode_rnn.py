
import torch
import torch.nn as nn

from torchdiffeq import odeint

from utils.linear import Linear
from crossbar.crossbar import crossbar

class ODE_Func(nn.Module):

    def __init__(self, hidden_layer_size, cb):
        super(ODE_Func, self).__init__()
        self.linear = Linear(hidden_layer_size, hidden_layer_size, cb)
        self.nonlinear = nn.Tanh()

    def forward(self, t, x):
        out = self.nonlinear(self.linear(x))

class ODE_RNN(nn.Module):

    def __init__(self, input_size, hidden_layer_size, output_size, device_params):
        
        super(ODE_RNN, self).__init__()

        # Initialize model instance crossbar
        self.cb = crossbar(device_params)

        # Initialize initial hidden state
        self.h_0 = torch.zeroes(hidden_layer_size, 1)

        # Model layers
        self.linear_in = Linear(input_size, hidden_layer_size, self.cb)
        self.linear_hidden = Linear(hidden_layer_size, hidden_layer_size, self.cb)
        self.decoder = Linear(hidden_layer_size, output_size, self.cb)

        self.ode_func = ODE_Func(hidden_layer_size, self.cb)
        self.nonlinear = nn.Tanh()

    def forward(self, t, x, method, step_size):
        
        h_i = self.h_0

        # RNN iteration
        for i, x_i in enumerate(x):
            h_i = odeint(self.ode_func, h_i, t.view(-1))
            h_i = self.nonlinear(self.linear_in(x_i) + self.linear_hidden(h_i))

        out = self.decoder(h_i)
        return out
