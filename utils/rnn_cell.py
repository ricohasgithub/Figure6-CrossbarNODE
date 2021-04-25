
import torch
import torch.nn as nn

from utils.linear import Linear
from crossbar import crossbar

class GRU_Cell(nn.Module):

    def __init__(self, input_layer_size, hidden_layer_size, cb):

        super(GRU_Cell, self).__init__()

        self.cb = cb

        self.i2h = Linear(input_layer_size, 3 * hidden_layer_size, cb)
        self.h2h = Linear(hidden_layer_size, 3 * hidden_layer_size, cb)

        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, h):

        gate_x = self.i2h(x).squeeze()
        gate_h = self.h2h(h).squeeze()

        i_r,i_u,i_n = gate_x.chunk(3, 1)
        h_r,h_u,h_n = gate_h.chunk(3, 1)
        
        gate_reset = torch.sigmoid(i_r + h_r)
        gate_update = torch.sigmoid(i_u + h_u)
        gate_nstate = torch.tanh(i_n + (gate_reset*h_n))

        return gate_update * h + (1 - gate_update) * gate_nstate

    def remap(self):
        self.i2h.remap()
        self.h2h.remap()
    
    def use_cb(self, state):
        self.i2h.use_cb(state)
        self.h2h.use_cb(state)