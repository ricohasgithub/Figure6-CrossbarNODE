"""
Louis Primeau
University of Toronto
Mar 2021
This python script outputs graphs for Figure 6 of the paper. This includes:
part a) Spiral datasets and predictions
"""

import torch
import networks.ode as ODE

import time
import random
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import numpy as np
import seaborn as sns

from torchdiffeq import odeint

def train(examples, model, epochs):
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_function = torch.nn.MSELoss()
    loss_history = []
    #model.remap("0")
    for epoch in range(epochs):
        #random.shuffle(examples)
        epoch_loss = []
        for i, (example, label) in enumerate(examples):
            
            optimizer.zero_grad()
            prediction = odeint(model, example)
            loss = loss_function(prediction, label)
            epoch_loss.append(loss)
            loss.backward()
            optimizer.step()
        
        loss_history.append(sum(epoch_loss) / len(examples))
        epoch_loss = []

    return loss_history

def test(seq, t, length, model):
    dt = torch.sum(t[1:] - t[0:-1]) / (len(t) - 1)
    print(dt, t[-1] - t[-2])
    output = []
    all_t = []
    with torch.no_grad():
        for i in range(length):
            prediction = model((seq, t + dt)).reshape(1, -1, 1)
            seq = torch.cat((seq[1:], prediction), axis=0)
            all_t.append(t[-1].unsqueeze(0) + dt.unsqueeze(0))
            t = torch.cat((t[1:], t[-1].unsqueeze(0) + dt.unsqueeze(0)), axis=0)
            output.append(prediction)

    return torch.cat(output, axis=0), torch.cat(all_t, axis=0)

pi = 3.14159265359

# DEVICE PARAMS for convenience.
device_params = {"Vdd": 1.8,
                 "r_wl": 20,
                 "r_bl": 20,
                 "m": 512,
                 "n": 512,
                 "r_on": 1e4,
                 "r_off": 1e5,
                 "dac_resolution": 4,
                 "adc_resolution": 14,
                 "bias_scheme": 1/3,
                 "tile_rows": 8,
                 "tile_cols": 8,
                 "r_cmos_line": 600,
                 "r_cmos_transistor": 20,
                 "p_stuck_on": 0.01,
                 "p_stuck_off": 0.01,
                 "method": "linear",
                 "r_on_mean": 1e4,
                 "r_on_stddev": 1e3,
                 "r_off_mean": 1e5,
                 "r_off_stddev": 1e4,
                 "device_resolution": 4,
}

# MAKE DATA

n_pts = 150 # How many data points to generate
size = 2 # 2D dataset
tw = 10 # Train window size
cutoff = 50 # Up to which data point is used for training

x = torch.linspace(0, 20, n_pts).reshape(1, -1)
y = torch.cat((torch.cos(x), torch.sin(x)), axis=0)
data = [((y[:, i:i+tw].reshape(-1, size, 1), x[:, i:i+tw].reshape(-1, 1, 1)), (y[:, i+tw:i+tw+1].reshape(size, -1))) for i in range(y.size(1) - tw)]
train_data, test_start = data[:cutoff], data[cutoff]

# CONFIGURE PLOTS
fig1 = plt.figure()
ax1 = plt.axes(projection='3d')

# TRAIN MODELS AND PLOT
time_steps = 30
epochs = 20
num_predict= 30
start_time = time.time()
for i in range(1):
    print("Model", i, "| elapsed time:", "{:5.2f}".format((time.time() - start_time) / 60), "min")
    model = rnn_ode.RNN_ODE(2, 6, 2, crossbar(device_params))
    losses = train(train_data, model, epochs)
    # model.node_rnn.observe(True)
    #model.use_cb(True)
    output, times = test(test_start[0][0], test_start[0][1], num_predict, model)
    # model.use_cb(False)

    o1, o2, o3 = output[:, 0].squeeze(), output[:, 1].squeeze(), times.squeeze()
    ax1.plot3D(o1, o2, o3, 'gray')
    
    d1, d2, d3 = y[0, :].squeeze(), y[1, :].squeeze(), x.squeeze()
    ax1.plot3D(d1, d2, d3, 'orange')
    
#fig1.savefig('output/fig6/1.png', dpi=600, transparent=True)
plt.show()