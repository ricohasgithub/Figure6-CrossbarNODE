
import time
import numpy as np

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
        self.linear2 = nn.Linear(hidden_layer_size, output_size)
        self.nonlinear = nn.Tanh()

    def forward(self, t, y):
        # y = torch.transpose(y, 0, 1)
        out = self.linear1(y)
        out = self.linear2(self.nonlinear(out))
        return out

    def remap(self):
        self.linear1.remap()
        self.linear2.remap()

    def use_cb(self, state):
        self.linear1.use_cb(state)
        self.linear2.use_cb(state)

def iter_train(model, data_gen, iters, method="dopri5", step_size="1"):

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

def epoch_train(model, data_gen, epochs):

    examples = data_gen.train_data

    optimizer = optim.Adam(model.parameters(), lr=0.01)
    loss_function = nn.MSELoss()

    loss_history = []

    for epoch in range(epochs):

        epoch_loss = []

        for i, (example, label) in enumerate(examples):
            
            optimizer.zero_grad()
            label = label.reshape(-1)
            prediction = odeint(model, example[0].reshape((10, 2)), example[1].reshape(-1))[0][0]

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
    
    with torch.no_grad():
        for i in range(length):
            #print(seq)
            #print(seq.size())
            prediction = odeint(model, seq, (t + dt).reshape(-1)).reshape(1, -1, 1)
            seq = torch.cat((seq[1:], prediction), axis=0)
            all_t.append(t[-1].unsqueeze(0) + dt.unsqueeze(0))
            t = torch.cat((t[1:], t[-1].unsqueeze(0) + dt.unsqueeze(0)), axis=0)
            output.append(prediction)

    output, times = torch.cat(output, axis=0), torch.cat(all_t, axis=0)

    o1, o2, o3 = output[:, 0].squeeze(), output[:, 1].squeeze(), times.squeeze()
    ax1.plot3D(o1, o2, o3, 'gray')
    
    d1, d2, d3 = data_gen.y[0, :].squeeze(), data_gen.y[1, :].squeeze(), data_gen.x.squeeze()
    ax1.plot3D(d1, d2, d3, 'orange')

    plt.show()