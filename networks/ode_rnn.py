
import torch
import torch.nn as nn
import torch.optim as optim

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
                # print(h_i.size())
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
