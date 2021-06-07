
import torch
import torch.nn as nn
import torch.optim as optim
import torch.functional as F

import random
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from mpl_toolkits import mplot3d
from matplotlib.lines import Line2D

from torchdiffeq import odeint
from crossbar.crossbar import crossbar

from utils.spiral_generator import Epoch_Spiral_Generator
from utils.spiral_generator import Epoch_Test_Spiral_Generator
from utils.spiral_generator import Stochastic_Spiral_Generator
from utils.spiral_generator import Regular_Spiral_Generator
from utils.spiral_generator import Epoch_AM_Wave_Generator
from utils.spiral_generator import Epoch_Heart_Generator

from networks.ode import ODE_Func as ODE_Net
from networks.ode import iter_train as iter_train

from networks.ode_rnn import ODE_RNN as ODE_RNN
from networks.ode_rnn import train as ode_rnn_train

from networks.ode_rnn_autogen import ODE_RNN as ODE_RNN_autogen
from networks.ode_rnn_autogen import train as ode_rnn_autogen_train

from networks.latent_ode import ODE_RNN as ODE_RNN_Test
from networks.latent_ode import train as ode_rnn_test_train

from networks.lstm_rnn import LSTM_RNN as LSTM_RNN
from networks.lstm_rnn import train as lstm_train

from networks.gru_rnn import GRU_RNN as GRU_RNN
from networks.gru_rnn import train as gru_train

# Color graphing utility
def random_color():
    rgb = [random.uniform(0.0, 1.0), random.uniform(0.0, 1.0), random.uniform(0.0, 1.0)]
    # rgb = [0.0, 0.0, 0.0]
    return tuple(rgb)

# Function to map and plot crossbar map for a given model
def plot_cmap(model):

    # Retrieve crossbar weights size
    weights = [model.cb.W[coord[0]:coord[0]+coord[2], coord[1]*2:coord[1]*2+coord[3]*2] for coord in model.cb.mapped] + [model.cb.W]
    vmax = max(torch.max(weight) for weight in weights)
    vmin = min(torch.min(weight) for weight in weights)

    # Plot crossbar mapping
    fig, ax_cmap = plt.subplots(ncols=len(weights), figsize=(20, 3))
    cmap = sns.blend_palette(("#fa7de3", "#ffffff", "#6ef3ff"), n_colors=9, as_cmap=True, input='hex')

    for ax in ax_cmap:
        ax.set(xticklabels=[])
        ax.set(yticklabels=[])
        
    with torch.no_grad():
        for i, weight in enumerate(weights):
            sns.heatmap(weight, vmax=vmax, vmin=vmin, cmap=cmap, square=True, cbar=False, ax=ax_cmap[i])

    fig.savefig("./output/ode_rnn_cmap.png", dpi=600, transparent=True)
    plt.close()

    return fig, ax_cmap

def plot_loss(epochs, loss):
    # Plot loss history
    fig, ax_loss = plt.subplots()
    fig.suptitle('ODE-RNN Error')
    ax_loss.plot(list(range(epochs)), loss, linewidth=1, color='c')
    fig.savefig('./output/training.png', dpi=600, transparent=True)
    plt.close()
    return fig, ax_loss

def build_model(epochs, data_gen, device_params, method, time_steps):

    # Build and train models
    # ode_rnn = ODE_RNN(2, 6, 2, device_params, method, time_steps)
    # losses_ode_rnn, output_ode_rnn = ode_rnn_train(ode_rnn, data_gen, epochs)
    ode_rnn = ODE_RNN_autogen(2, 6, 2, device_params, method, time_steps)
    losses_ode_rnn, output_ode_rnn = ode_rnn_autogen_train(ode_rnn, data_gen, epochs)

    # Plot crossbar mapping and loss
    fig_cmap, ax_cmap = plot_cmap(ode_rnn)
    fig_loss, ax_loss = plot_loss(epochs, losses_ode_rnn)

    return ode_rnn, output_ode_rnn, losses_ode_rnn

def get_average_performance(iters, epochs, device_params, method, time_steps):

    # Get regular spiral data with irregularly sampled time intervals (+ noise)
    # data_gen = Epoch_Spiral_Generator(80, 20, 40, 20, 2, 79)
    # data_gen = Epoch_Test_Spiral_Generator(80, 40, 20, 10, 2)

    loss_avg = [0] * epochs
    loss_history = []

    for i in range(iters):

        # Get current model output
        model, output, loss = build_model(epochs, data_gen, device_params, method, time_steps)
        loss_history.append(loss)

        for j in range(len(loss)):
            loss_avg[j] += loss[j]

        print('Iter {:04d}'.format(i))

    for i in range(len(loss_avg)):
        loss_avg[i] = (loss_avg[i]/iters)

    return loss_avg

def graph_average_performance(iters, epochs, device_params, method, time_steps):

    # Get regular spiral data with irregularly sampled time intervals (+ noise)
    # data_gen = Epoch_Spiral_Generator(80, 20, 40, 20, 2, 79)
    # data_gen = Epoch_Test_Spiral_Generator(80, 40, 20, 10, 2)
    # data_gen = Epoch_AM_Wave_Generator(80, 20, 40, 10, 2)
    data_gen = Epoch_Heart_Generator(80, 20, 40, 10, 2)

    ax = plt.axes(projection='3d')
    loss_avg = [0] * epochs
    loss_history = []

    colors = []

    for i in range(iters):
        colors.append(random_color())

    for i in range(iters):

        # Get current model output
        model, output, loss = build_model(epochs, data_gen, device_params, method, time_steps)
        loss_history.append(loss)

        ax.plot3D(output[0], output[1], output[2], color=colors[i], linewidth=1.5)
        # ax.scatter3D(output[0], output[1], output[2], color='c')

        for j in range(len(loss)):
            loss_avg[j] += loss[j]

        print('Iter {:04d}'.format(i))

    for i in range(len(loss_avg)):
        loss_avg[i] = (loss_avg[i]/iters)

    # Plot true trajectory and observation points
    d1, d2, d3 = data_gen.y[0, :].squeeze(), data_gen.y[1, :].squeeze(), data_gen.x.squeeze()
    ax.plot3D(data_gen.true_x, data_gen.true_y, data_gen.true_z, 'gray')
    ax.scatter3D(d1, d2, d3, 'gray')
    plt.savefig('./output/ode_rnn.png', dpi=600, transparent=True)

    # Plot loss history and average loss
    fig, ax_loss = plt.subplots()
    fig.suptitle('Average ODE-RNN Error')

    for i in range(iters):
        ax_loss.plot(list(range(epochs)), loss_history[i], color=colors[i], linewidth=1)        

    ax_loss.plot(list(range(epochs)), loss_avg, color='black', linewidth=1)
    fig.savefig('./output/training_avg.png', dpi=600, transparent=True)

    return fig, ax_loss, loss_avg

def graph_ode_solver_difference(iters, epochs, device_params):

    # List of ODE Solver Functions
    fixed_step_methods = ["euler", "midpoint", "rk4", "explicit_adams", "implicit_adams"]
    adaptive_step_methods = ["dopri8", "dopri5", "bosh3", "fehlberg2", "adaptive_heun"]

    colors = []

    for i in range(len(fixed_step_methods)):
        colors.append(random_color())

    fig, ax = plt.subplots()
    all_loss = []

    for i in range(len(fixed_step_methods)):
        print("NOW USING: ", fixed_step_methods[i])
        loss_avg = get_average_performance(iters, epochs, device_params, fixed_step_methods[i], 1)
        ax.plot(list(range(epochs)), loss_avg, colors[i], linewidth=1.5)
        all_loss.append(Line2D([0], [0], color=colors[i], lw=4))

    fig.savefig('./output/ode_solver_difference.png', dpi=600, transparent=True)

    ax.legend(all_loss, fixed_step_methods)

    return fig, ax

# Device parameters for convenience
device_params = {"Vdd": 0.2,
                 "r_wl": 20,
                 "r_bl": 20,
                 "m": 64,
                 "n": 64,
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

graph_average_performance(1, 100, device_params, "euler", 1)
# graph_ode_solver_difference(10, 30, device_params)

# data_gen = Epoch_AM_Wave_Generator(80, 20, 40, 20, 2)

# ax = plt.axes(projection='3d')
# ax.plot3D(data_gen.y_x.squeeze(), data_gen.y_y.squeeze(), data_gen.x.squeeze(), 'gray')
# ax.scatter3D(data_gen.y_x.squeeze(), data_gen.y_y.squeeze(), data_gen.x.squeeze(), 'blue')

# ode_rnn = GRU_RNN(2, 6, 2, device_params)
# losses_ode_rnn, output_ode_rnn = gru_train(ode_rnn, data_gen, epochs)

# data_gen = Epoch_Test_Spiral_Generator(80, 40, 20, 10, 2)
# data_gen = Epoch_Test_Spiral_Generator(80, 20, 40, 20, 2)

# ode_rnn_autogen = ODE_RNN_autogen(2, 6, 2, device_params)
# losses_ode_rnn, output_ode_rnn = ode_rnn_autogen_train(ode_rnn_autogen, data_gen, 30)

# ode_rnn = ODE_RNN_Test(2, 6, 2, device_params)
# losses_ode_rnn, output_ode_rnn = ode_rnn_test_train(ode_rnn, data_gen, 30)

# lstm_rnn = LSTM_RNN(2, 6, 2, device_params)
# output_lstm = lstm_train(lstm_rnn, data_gen, 100)

# ode_net = ODE_Net(3, 50, 3, crossbar(device_params))
# iter_train(ode_net, data_gen2, 500)

plt.show()
