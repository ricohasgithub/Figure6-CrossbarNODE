
import torch
import torch.nn as nn

from torchdiffeq import odeint

from utils.linear import Linear
from crossbar.crossbar import crossbar

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