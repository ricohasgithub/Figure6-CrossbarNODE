
import torch
import torch.nn as nn
import torch.optim as optim

from torchdiffeq import odeint

from crossbar.crossbar import crossbar

from utils.linear import Linear
from utils.running_avg_meter import Running_Average_Meter

class ODE_Func(nn.Module):

    def __init__(self, input_size, hidden_layer_size, output_size, cb):

        super(ODE_Func, self).__init__()

        # Model layers
        self.linear1 = Linear(input_size, hidden_layer_size, cb)
        self.linear2 = Linear(hidden_layer_size, output_size, cb)
        self.nonlinear = nn.Tanh()

    def forward(self, t, y):
        out = self.linear1(y**3)
        out = self.nonlinear(out)
        out = self.linear2(out)
        return out

    def remap(self):
        self.linear1.remap()
        self.linear2.remap()

    def use_cb(self, state):
        self.linear1.use_cb(state)
        self.linear2.use_cb(state)

def train(ode_model, train_data, test_data, epochs, method, step_size):
    pass