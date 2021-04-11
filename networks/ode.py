
import torch
import torch.nn as nn

from utils.linear import Linear
from crossbar.crossbar import crossbar

class ODE_Func(nn.Module):

    def __init__(self, params):
        super(ODE_Func, self).__init__()
