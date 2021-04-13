
import time

import torch
import torch.nn as nn
import torch.optim as optim

from torchdiffeq import odeint
from crossbar.crossbar import crossbar
from utils.linear import Linear

class ODE_Func(nn.Module):

    def __init__(self, input_size, hidden_layer_size, output_size, cb):

        super(ODE_Func, self).__init__()

        # Model layers
        self.linear1 = nn.Linear(input_size, hidden_layer_size)
        self.linear2 = nn.Linear(hidden_layer_size, hidden_layer_size)
        self.linear3 = nn.Linear(hidden_layer_size, output_size)
        self.nonlinear = nn.Tanh()

    def forward(self, t, y):
        # y = torch.transpose(y, 0, 1)
        # print(y.size())
        # out = self.nonlinear(self.linear1(y))
        # out = self.nonlinear(self.linear2(out))
        # out = self.linear3(out)
        out = self.linear1(y)
        out = self.linear2(self.nonlinear(out))
        out = self.linear3(self.nonlinear(out))
        return out

    def remap(self):
        self.linear1.remap()
        self.linear2.remap()

    def use_cb(self, state):
        self.linear1.use_cb(state)
        self.linear2.use_cb(state)

def train(model, data_gen, iters, method="dopri5", step_size="1"):

    #ode_model.use_cb(True)
    
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    for itr in range(1, iters + 1):

        # Prepare batch
        optimizer.zero_grad()
        batch_y0, batch_t, batch_y = data_gen.get_random_batch()
        
        # Make prediction
        pred_y = odeint(model, batch_y0, batch_t)

        # Calculate loss and backprop
        loss = torch.mean(torch.abs(pred_y - batch_y))
        loss.backward()
        optimizer.step()

        print('Iter {:04d} | Total Loss {:.6f}'.format(itr, loss.item()))

        # if itr%20 == 0:

        #     with torch.no_grad():

        #         pred_y = odeint(model, data_gen.true_y0, data_gen.t)

        #         loss = torch.mean(torch.abs(pred_y - data_gen.true_y))
        #         data_gen.plot_prediction(data_gen.true_y, pred_y)

    with torch.no_grad():

        pred_y = odeint(model, data_gen.true_y0, data_gen.t)

        loss = torch.mean(torch.abs(pred_y - data_gen.true_y))
        data_gen.plot_prediction(data_gen.true_y, pred_y)