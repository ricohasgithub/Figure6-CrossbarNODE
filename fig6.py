
import torch
import torch.nn as nn
import torch.optim as optim
import torch.functional as F

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from mpl_toolkits import mplot3d

from torchdiffeq import odeint
from crossbar.crossbar import crossbar

from utils.spiral_generator import Epoch_Spiral_Generator
from utils.spiral_generator import Stochastic_Spiral_Generator
from utils.spiral_generator import Regular_Spiral_Generator

from networks.ode import ODE_Func as ODE_Net
from networks.ode import iter_train as iter_train

from networks.ode_rnn import ODE_RNN as ODE_RNN
from networks.ode_rnn import train as ode_rnn_train

from networks.latent_ode import ODE_RNN as ODE_RNN_Test
from networks.latent_ode import train as ode_rnn_test_train

from networks.lstm_rnn import LSTM_RNN as LSTM_RNN
from networks.lstm_rnn import train as lstm_train

from networks.gru_rnn import GRU_RNN as GRU_RNN
from networks.gru_rnn import train as gru_train

# Function to map and plot crossbar map for a given model
def plot_cmap(model):

    # Plot crossbar mapping
    fig, ax_cmap = plt.subplots(ncols=5, figsize=(20, 3))
    cmap = sns.blend_palette(("#fa7de3", "#ffffff", "#6ef3ff"), n_colors=9, as_cmap=True, input='hex')

    for ax in ax_cmap:
        ax.set(xticklabels=[])
        ax.set(yticklabels=[])

    weights = [model.cb.W[coord[0]:coord[0]+coord[2], coord[1]*2:coord[1]*2+coord[3]*2] for coord in model.cb.mapped] + [model.cb.W]
    vmax = max(torch.max(weight) for weight in weights)
    vmin = min(torch.min(weight) for weight in weights)

    with torch.no_grad():
        for i, weight in enumerate(weights):
            sns.heatmap(weight, vmax=vmax, vmin=vmin, cmap=cmap, square=True, cbar=False, ax=ax_cmap[i])

    plt.savefig()

    return fig, ax_cmap


# Device parameters for convenience     
device_params = {"Vdd": 0.2,
                 "r_wl": 20,
                 "r_bl": 20,
                 "m": 128,
                 "n": 128,
                 "r_on": 1e4,
                 "r_off": 1e5,
                 "dac_resolution": 4,
                 "adc_resolution": 14,
                 "bias_scheme": 1/3,
                 "tile_rows": 8,
                 "tile_cols": 8,
                 "r_cmos_line": 600,
                 "r_cmos_transistor": 20,
                 "r_on_stddev": 1e3,
                 "r_off_stddev": 1e4,
                 "p_stuck_on": 0.01,
                 "p_stuck_off": 0.01,
                 "method": "viability",
                 "viability": 0.05,
}

# Get regular spiral data with irregularly sampled time intervals (+ noise)
# data_gen = Regular_Spiral_Generator(50, 50, 25, 10, 20)
# ground_truth = data_gen.get_plot()
# plt.savefig('./output/ground_truth.png', dpi=600, transparent=True)

# 40, 10, 20, 10, 2
data_gen = Epoch_Spiral_Generator(40, 20, 20, 10, 2)

ax = plt.axes(projection='3d')

# Build and train models

epochs = 30

ode_rnn = GRU_RNN(2, 6, 2, device_params)
losses_ode_rnn, output_ode_rnn = gru_train(ode_rnn, data_gen, epochs)

# ode_rnn = ODE_RNN_Test(2, 6, 2, device_params)
# losses_ode_rnn, output_ode_rnn = ode_rnn_test_train(ode_rnn, data_gen, epochs)

# ode_rnn = ODE_RNN(2, 6, 2, device_params)
# losses_ode_rnn, output_ode_rnn = ode_rnn_train(ode_rnn, data_gen, epochs)

# lstm_rnn = LSTM_RNN(2, 6, 2, device_params)
# output_lstm = lstm_train(lstm_rnn, data_gen, 100)

# ode_net = ODE_Net(3, 50, 3, crossbar(device_params))
# iter_train(ode_net, data_gen2, 500)

# Test models

# Display all remaining plots
# plt.setup(output_ode_rnn)
# plt.setup(output_lstm)

# Plot loss history
fig1, ax_loss = plt.subplots()
fig1.suptitle('ODE-RNN Error')
ax_loss.plot(list(range(epochs)), losses_ode_rnn, linewidth=1, marker = 's', color='c')

plt.show()
