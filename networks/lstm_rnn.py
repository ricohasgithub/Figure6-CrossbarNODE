
import torch
import torch.nn as nn
import torch.optim as optim

import matplotlib.pyplot as plt

from utils.linear import Linear
from crossbar.crossbar import crossbar

class LSTM_RNN(nn.Module):

    def __init__(self, input_layer_size, hidden_layer_size, output_layer_size, device_params):

        super(LSTM_RNN, self).__init__()

        self.input_layer_size = input_layer_size
        self.hidden_layer_size = hidden_layer_size
        self.output_layer_size = output_layer_size

        self.cb = crossbar(device_params)

        self.lstm = nn.LSTM(input_layer_size, hidden_layer_size, num_layers=1, batch_first=True)
        self.linear = nn.Linear(hidden_layer_size, output_layer_size)
    
    def forward(self, x):
        out, h = self.lstm(x)
        out = out[:, 1, :]
        out = self.linear(torch.relu(out))
        return out

    def use_cb(self, state):
        self.linear_in.use_cb(state)
        self.linear_hidden.use_cb(state)
        self.solve.use_cb(state)

def train(model, data_gen, epochs):

    examples = data_gen.train_data

    optimizer = optim.Adam(model.parameters(), lr=0.01)
    loss_function = nn.MSELoss()

    loss_history = []

    for epoch in range(epochs):

        epoch_loss = []

        for i, (example, label) in enumerate(examples):

            optimizer.zero_grad()
            prediction = model(example[0].reshape(-1, 10, 2)).reshape(2, 1)

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
    num_predict = 30
    length = num_predict

    dt = torch.sum(t[1:] - t[0:-1]) / (len(t) - 1)
    output = []
    all_t = []

    #model.use_cb(True)
    
    with torch.no_grad():
        for i in range(length):
            prediction = model(seq.reshape(-1, 10, 2)).reshape(2, 1).reshape(1, -1, 1)
            seq = torch.cat((seq[1:], prediction), axis=0)
            all_t.append(t[-1].unsqueeze(0) + dt.unsqueeze(0))
            t = torch.cat((t[1:], t[-1].unsqueeze(0) + dt.unsqueeze(0)), axis=0)
            output.append(prediction)

    output, times = torch.cat(output, axis=0), torch.cat(all_t, axis=0)

    ax = plt.axes(projection='3d')

    o1, o2, o3 = output[:, 0].squeeze(), output[:, 1].squeeze(), times.squeeze()
    # o1, o2, o3 = output[:, 0].squeeze(), output[:, 1].squeeze(), output[:, 2].squeeze()
    ax.plot3D(o1, o2, o3, 'red')
    ax.scatter3D(o1, o2, o3, 'red')
    
    d1, d2, d3 = data_gen.y[0, :].squeeze(), data_gen.y[1, :].squeeze(), data_gen.x.squeeze()
    # d1, d2, d3 = data_gen.y[0, :].squeeze(), data_gen.y[1, :].squeeze(), data_gen.y[2, :].squeeze()
    # ax.plot3D(d1, d2, d3, 'gray')
    ax.plot3D(data_gen.true_x, data_gen.true_y, data_gen.true_z, 'gray')
    ax.scatter3D(d1, d2, d3, 'gray')

    plt.savefig('./output/ode_rnn.png', dpi=600, transparent=True)

    return ax

