
import torch
import torch.nn as nn
import torch.optim as optim

import random
import matplotlib.pyplot as plt

from utils.linear import Linear
from utils.rnn_cell import GRU_Cell
from crossbar.crossbar import crossbar

class GRU_RNN(nn.Module):

    def __init__(self, input_size, hidden_layer_size, output_size, device_params):

        super(GRU_RNN, self).__init__()

        self.input_size = input_size
        self.hidden_layer_size = hidden_layer_size
        self.output_size = output_size

        self.cb = crossbar(device_params)

        self.rnn_cell = GRU_Cell(input_size, hidden_layer_size, self.cb)
        self.linear = Linear(hidden_layer_size, output_size, self.cb)
    
    def forward(self, t, x, method="dopri5", step_size=20):

        t = t.reshape(-1).float()
        N = t.shape[0]

        # Model current output hidden state dynamics
        h_i = torch.zeros(self.hidden_layer_size, 1)
        output = torch.zeros(N, self.output_size)

        # Initial layer (h0)
        if t[0] > 0:

            h_i = self.rnn_cell(x[0], h_i)
            h_i = torch.transpose(h_i, 0, 1)
            out = self.linear(torch.relu(h_i))
            output[0] = out.reshape(-1)

        for i in range(1, N):

            h_i = self.rnn_cell(x[0], h_i)
            h_i = torch.transpose(h_i, 0, 1)
            out = self.linear(torch.relu(h_i))
            
            output[i] = out.reshape(-1)

        return output

    def use_cb(self, state):
        self.linear_in.use_cb(state)
        self.linear_hidden.use_cb(state)
        self.solve.use_cb(state)

def train(model, data_gen, epochs):

    #model.use_cb(True)

    examples = data_gen.train_data

    optimizer = optim.Adam(model.parameters(), lr=0.01)
    loss_function = nn.MSELoss()

    loss_history = []

    for epoch in range(epochs):

        random.shuffle(examples)
        epoch_loss = []

        for i, (example, label) in enumerate(examples):
            
            optimizer.zero_grad()
            # Make example[1] more than just 1 point to have model output predict sequence
            prediction = model(example[1], example[0]).transpose(0, 1)

            # print("pred: ", prediction)
            # print("label: ", label)

            loss = loss_function(prediction, label)
            epoch_loss.append(loss)
            loss.backward()
            optimizer.step()
        
        loss_history.append(sum(epoch_loss) / len(examples))
        epoch_loss = []

        print('Epoch {:04d} | Total Loss {:.6f}'.format(epoch, loss_history[epoch]))

    # Test
    # seq = data_gen.test_start[0][0]
    seq = data_gen.test_data[0][0][0]
    print("seq: ", seq.size())
    print(seq)
    #t = data_gen.test_start[0][1]
    times = data_gen.test_data[0][0][1]
    # num_predict currently not being used
    num_predict = 50
    length = num_predict

    # dt = torch.sum(t[1:] - t[0:-1]) / (len(t) - 1)
    output = []
    # all_t = []

    # model.use_cb(True)

    # for i in range(num_predict):
    #     all_t.append(t[-1].unsqueeze(0) + dt.unsqueeze(0))
    #     t = torch.cat((t[1:], t[-1].unsqueeze(0) + dt.unsqueeze(0)), axis=0)
    
    # times = torch.cat(all_t, axis=0)

    # TODO: make seq same length as times (30)

    with torch.no_grad():
        prediction = model(times, seq)
        print("pred2: ", prediction.size())
        print(prediction)
        output.append(prediction)
    
    output = torch.cat(output, axis=0)
    
    # with torch.no_grad():
    #     for i in range(length):
    #         prediction = model((t + dt), seq).reshape(1, -1, 1)
    #         seq = torch.cat((seq[1:], prediction), axis=0)
    #         all_t.append(t[-1].unsqueeze(0) + dt.unsqueeze(0))
    #         t = torch.cat((t[1:], t[-1].unsqueeze(0) + dt.unsqueeze(0)), axis=0)
    #         output.append(prediction)

    # output, times = torch.cat(output, axis=0), torch.cat(all_t, axis=0)

    ax = plt.axes(projection='3d')

    o1, o2, o3 = output[:, 0].squeeze(), output[:, 1].squeeze(), times.squeeze()
    # o1, o2, o3 = output[:, 0].squeeze(), output[:, 1].squeeze(), output[:, 2].squeeze()
    ax.plot3D(o1, o2, o3, 'red')
    # ax.scatter3D(o1, o2, o3, 'red')
    
    d1, d2, d3 = data_gen.y[0, :].squeeze(), data_gen.y[1, :].squeeze(), data_gen.x.squeeze()
    # d1, d2, d3 = data_gen.y[0, :].squeeze(), data_gen.y[1, :].squeeze(), data_gen.y[2, :].squeeze()
    # ax.plot3D(d1, d2, d3, 'gray')
    ax.plot3D(data_gen.true_x, data_gen.true_y, data_gen.true_z, 'gray')
    ax.scatter3D(d1, d2, d3, 'gray')

    plt.savefig('./output/gru_rnn.png', dpi=600, transparent=True)

    return loss_history, ax