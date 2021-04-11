
import torch
import torch.nn as nn

from utils.linear import Linear
from crossbar.crossbar import crossbar

class LSTM_RNN(nn.Module):

    def __init__(self, params):

        super(LSTM_RNN, self).__init__()

        