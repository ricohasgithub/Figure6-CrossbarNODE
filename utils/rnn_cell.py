
import torch
import torch.nn as nn

from utils.linear import Linear
from crossbar import crossbar

class GRU_Cell(nn.Module):

    def __init__(self, io_layer_size, hidden_layer_size, gate_out_size):

        super(GRU_Cell, self).__init__()

        # Update gate
        self.update_lin1 = Linear(io_layer_size + 1, hidden_layer_size)
        self.update_lin2 = Linear(hidden_layer_size, gate_out_size)

        # Reset gate
        self.reset_lin1 = Linear(io_layer_size + 1, hidden_layer_size)
        self.reset_lin2 = Linear(hidden_layer_size, gate_out_size)

        # New state gate
        self.nstate_lin1 = Linear(io_layer_size + 1, hidden_layer_size)
        self.nstate_lin2 = Linear(hidden_layer_size, io_layer_size)

        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, h):

        

        pass