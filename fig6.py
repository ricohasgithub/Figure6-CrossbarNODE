
'''
TODO:
    - Hardware related graphs: currents, mapping, crossbar variability
    - Performance graphs: different solvers, different time meshes
'''

import torch
import torch.nn as nn
import torch.optim as optim
import torch.functional as F

import copy
import random
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from mpl_toolkits import mplot3d
from matplotlib.lines import Line2D
from celluloid import Camera

from torchdiffeq import odeint
from crossbar.crossbar import crossbar

from utils.spiral_generator import Epoch_Spiral_Generator
from utils.spiral_generator import Epoch_Test_Spiral_Generator
from utils.spiral_generator import Stochastic_Spiral_Generator
from utils.spiral_generator import Regular_Spiral_Generator
from utils.spiral_generator import Epoch_AM_Wave_Generator
from utils.spiral_generator import Epoch_Heart_Generator
from utils.spiral_generator import Epoch_Spiral_Generator
from utils.spiral_generator import Epoch_Square_Generator
from utils.spiral_generator import Epoch_Noise_Spiral_Generator

from networks.ode import ODE_Func as ODE_Net
from networks.ode import iter_train as iter_train

from networks.ode_rnn import ODE_RNN as ODE_RNN
from networks.ode_rnn import train as ode_rnn_train

from networks.ode_rnn_autogen import ODE_RNN as ODE_RNN_autogen
from networks.ode_rnn_autogen import train as ode_rnn_autogen_train
from networks.ode_rnn_autogen import test as ode_rnn_autogen_test

from networks.latent_ode import ODE_RNN as ODE_RNN_Test
from networks.latent_ode import train as ode_rnn_test_train

from networks.lstm_rnn import LSTM_RNN as LSTM_RNN
from networks.lstm_rnn import train as lstm_train

from networks.gru_rnn import GRU_RNN as GRU_RNN
from networks.gru_rnn import train as gru_train

from networks.gru_rnn_autogen import GRU_RNN as GRU_RNN_autogen
from networks.gru_rnn_autogen import train as gru_rnn_autogen_train
from networks.gru_rnn_autogen import test as gru_rnn_autogen_test

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

def plot_currents(model):
    print(len(model.cb.current_history))

def plot_loss(epochs, loss):
    # Plot loss history
    fig, ax_loss = plt.subplots()
    fig.suptitle('ODE-RNN Error')
    ax_loss.plot(list(range(epochs)), loss, linewidth=1, color='c')
    fig.savefig('./output/training.png', dpi=600, transparent=True)
    plt.close()
    return fig, ax_loss

def animate_model_output(fig, ax, data_gen, color, output):

    camera = Camera(fig)

    d1, d2, d3 = data_gen.y[0, :].squeeze(), data_gen.y[1, :].squeeze(), data_gen.x.squeeze()
    ax.plot3D(data_gen.true_x, data_gen.true_y, data_gen.true_z, 'gray')
    ax.scatter3D(d1, d2, d3, 'gray')
    
    for j in range(output[2].size()[0]):
        d1, d2, d3 = data_gen.y[0, :].squeeze(), data_gen.y[1, :].squeeze(), data_gen.x.squeeze()
        ax.plot3D(data_gen.true_x, data_gen.true_y, data_gen.true_z, 'gray')
        ax.scatter3D(d1, d2, d3, 'blue')
        ax.plot3D(output[0][:j], output[1][:j], output[2][:j], color=color, linewidth=1.5)
        camera.snap()
        plt.pause(0.02)

    animation = camera.animate()
    animation.save('output/animation.gif', writer='PillowWriter', fps=10)

def build_model(epochs, data_gen, device_params, method, time_steps):

    # Build and train models
    # ode_rnn = ODE_RNN(2, 6, 2, device_params, method, time_steps)
    # losses_ode_rnn, output_ode_rnn = ode_rnn_train(ode_rnn, data_gen, epochs)
    ode_rnn = ODE_RNN_autogen(2, 6, 2, device_params, method, time_steps)
    losses_ode_rnn = ode_rnn_autogen_train(ode_rnn, data_gen, epochs)
    output_ode_rnn = ode_rnn_autogen_test(ode_rnn, data_gen)

    # Plot crossbar mapping and loss
    fig_cmap, ax_cmap = plot_cmap(ode_rnn)
    fig_loss, ax_loss = plot_loss(epochs, losses_ode_rnn)

    return ode_rnn, output_ode_rnn, losses_ode_rnn

def get_average_gru_performance(iters, epochs, device_params):

    data_gen = Epoch_Test_Spiral_Generator(80, 40, 20, 10, 2)

    model = GRU_RNN_autogen(2, 6, 2, device_params)
    losses_gru, output = gru_rnn_autogen_train(model, data_gen, epochs)

    ax = plt.axes(projection='3d')
    ax.plot3D(output[0], output[1], output[2], color="black", linewidth=1.5)

    d1, d2, d3 = data_gen.y[0, :].squeeze(), data_gen.y[1, :].squeeze(), data_gen.x.squeeze()
    ax.plot3D(data_gen.true_x, data_gen.true_y, data_gen.true_z, 'gray')
    ax.scatter3D(d1, d2, d3, 'gray')

    return losses_gru, output

def get_average_gru_performance_datagen(iters, epochs, device_params, data_gen):

    model = GRU_RNN_autogen(2, 6, 2, device_params)
    losses_gru, output = gru_rnn_autogen_train(model, data_gen, epochs)

    return losses_gru

def get_average_performance_datagen(iters, epochs, data_gen, device_params, method, time_steps):

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

def get_average_performance(iters, epochs, device_params, method, time_steps):

    # Get regular spiral data with irregularly sampled time intervals (+ noise)
    # data_gen = Epoch_Spiral_Generator(80, 20, 40, 20, 2, 79)
    data_gen = Epoch_Test_Spiral_Generator(80, 40, 20, 10, 2)

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
    # data_gen = Epoch_Heart_Generator(160, 20, 40, 10, 2)
    # data_gen = Epoch_Spiral_Generator(80, 80, 40, 10, 2)
    data_gen = Epoch_Test_Spiral_Generator(80, 20, 40, 10, 2)

    fig = plt.figure()
    ax = plt.axes(projection='3d')

    # Plot true trajectory and observation points
    d1, d2, d3 = data_gen.y[0, :].squeeze(), data_gen.y[1, :].squeeze(), data_gen.x.squeeze()
    ax.plot3D(data_gen.true_x, data_gen.true_y, data_gen.true_z, 'gray')
    ax.scatter3D(d1, d2, d3, 'gray')

    loss_avg = [0] * epochs
    loss_history = []

    colors = []

    for i in range(iters):
        colors.append(random_color())

    for i in range(iters):

        # Get current model output
        model, output, loss = build_model(epochs, data_gen, device_params, method, time_steps)
        loss_history.append(loss)

        # animate_model_output(fig, ax, data_gen, colors[i], output)

        # d1, d2, d3 = data_gen.y[0, :].squeeze(), data_gen.y[1, :].squeeze(), data_gen.x.squeeze()
        # ax.plot3D(data_gen.true_x, data_gen.true_y, data_gen.true_z, 'gray')
        # ax.scatter3D(d1, d2, d3, 'gray')

        ax.plot3D(output[0], output[1], output[2], color=colors[i], linewidth=1.5)
        # ax.scatter3D(output[0], output[1], output[2], color='c')
        plot_currents(model)

        for j in range(len(loss)):
            loss_avg[j] += loss[j]

        print('Iter {:04d}'.format(i))

    for i in range(len(loss_avg)):
        loss_avg[i] = (loss_avg[i]/iters)

    plt.savefig('./output/ode_rnn.png', dpi=600, transparent=True)

    # Plot loss history and average loss
    fig, ax_loss = plt.subplots()
    fig.suptitle('Average ODE-RNN Error')

    for i in range(iters):
        ax_loss.plot(list(range(epochs)), loss_history[i], color=colors[i], linewidth=1)        

    ax_loss.plot(list(range(epochs)), loss_avg, color='black', linewidth=1)
    fig.savefig('./output/training_avg.png', dpi=600, transparent=True)

    return fig, ax_loss, loss_avg

def graph_model_difference(iters, epochs, device_params, method, time_steps):

    data_gen = Epoch_Test_Spiral_Generator(80, 20, 40, 10, 2)

    # Get ODE-RNN performance
    loss_ode = get_average_performance_datagen(iters, epochs, data_gen, device_params, method, time_steps)
    # Get GRU-RNN performance
    loss_gru = get_average_gru_performance_datagen(iters, epochs, device_params, data_gen)

    # Plot loss history and average loss
    fig, ax_loss = plt.subplots()
    fig.suptitle('Average MSE Loss')

    ax_loss.plot(list(range(epochs)), loss_ode, color="blue", linewidth=1)
    ax_loss.plot(list(range(epochs)), loss_gru, color="red", linewidth=1)

    all_loss = []
    all_loss.append(Line2D([0], [0], color="blue", lw=4))
    all_loss.append(Line2D([0], [0], color="red", lw=4))

    ax_loss.legend(all_loss, ["ODE-RNN", "GRU-RNN"])

    fig.savefig('./output/model_training_difference.png', dpi=600, transparent=True)

def graph_ode_solver_difference(iters, epochs, device_params):

    # List of ODE Solver Functions
    fixed_step_methods = ["euler", "midpoint", "rk4", "explicit_adams", "implicit_adams"]
    adaptive_step_methods = ["dopri8", "dopri5", "bosh3", "fehlberg2", "adaptive_heun"]

    colors = ["maroon", "goldenrod", "limegreen", "teal", "darkviolet"]

    ax = plt.axes(projection='3d')

    loss_fig, loss_ax = plt.subplots()
    all_loss = []

    data_gen = Epoch_Test_Spiral_Generator(80, 40, 20, 10, 2)

    for i in range(len(fixed_step_methods)):

        print("NOW USING: ", fixed_step_methods[i])

        model, output, loss = build_model(epochs, data_gen, device_params, fixed_step_methods[i], 1)
        ax.plot3D(output[0], output[1], output[2], color=colors[i], linewidth=1.5)
        loss_ax.plot(list(range(epochs)), loss, colors[i], linewidth=1.5)

        # loss_avg = get_average_performance(iters, epochs, device_params, fixed_step_methods[i], 1)
        # ax.plot(list(range(epochs)), loss_avg, colors[i], linewidth=1.5)

        all_loss.append(Line2D([0], [0], color=colors[i], lw=4))

    loss_fig.savefig('./output/ode_solver_difference.png', dpi=600, transparent=True)
    loss_ax.legend(all_loss, fixed_step_methods)
    ax.legend(all_loss, fixed_step_methods)

    # Plot true trajectory and observation points
    d1, d2, d3 = data_gen.y[0, :].squeeze(), data_gen.y[1, :].squeeze(), data_gen.x.squeeze()
    ax.plot3D(data_gen.true_x, data_gen.true_y, data_gen.true_z, 'gray')
    ax.scatter3D(d1, d2, d3, 'gray')

    plt.savefig('./output/ode_rnn.png', dpi=600, transparent=True)

    return ax, loss_fig, loss_ax

def graph_step_size_difference(iters, epochs, method, device_params, data_gen):

    fixed_step_sizes = [1e-2, 1e-1, 1, 10, 100, 1000]
    colors = ["maroon", "goldenrod", "limegreen", "teal", "darkviolet", "black"]
 
    ode_rnn_fig, ode_rnn_axs = plt.subplots(subplot_kw=dict(projection='3d'))

    loss_fig, loss_ax = plt.subplots()
    all_loss = []

    for i in range(len(fixed_step_sizes)):

        print("NOW USING: ", fixed_step_sizes[i])

        model, output, loss = build_model(epochs, data_gen, device_params, method, fixed_step_sizes[i])
        ode_rnn_axs.plot3D(output[0], output[1], output[2], color=colors[i], linewidth=1.5)
        loss_ax.plot(list(range(epochs)), loss, colors[i], linewidth=1.5)

        # loss_avg = get_average_performance(iters, epochs, device_params, fixed_step_methods[i], 1)
        # ax.plot(list(range(epochs)), loss_avg, colors[i], linewidth=1.5)

        all_loss.append(Line2D([0], [0], color=colors[i], lw=2))

    plt.xlabel("Epoch")
    plt.ylabel("MSE loss on +" + str(device_params["viability"]) + " variability crossbar")

    loss_fig.savefig('./output/ode_step_difference.png', dpi=600, transparent=True)
    loss_ax.legend(all_loss, fixed_step_sizes)
    ode_rnn_axs.legend(all_loss, fixed_step_sizes)

    # Plot true trajectory and observation points
    d1, d2, d3 = data_gen.y[0, :].squeeze(), data_gen.y[1, :].squeeze(), data_gen.x.squeeze()
    ode_rnn_axs.plot3D(data_gen.true_x, data_gen.true_y, data_gen.true_z, 'gray')
    ode_rnn_axs.scatter3D(d1, d2, d3, 'gray')
    ode_rnn_fig.suptitle("Time mesh differences on ODE-RNN")

    plt.savefig('./output/ode_rnn.png', dpi=600, transparent=True)

    return ode_rnn_axs, loss_fig, loss_ax

def single_model_plot(epochs, device_params, method, time_steps):

    # Create data generators with different nosie amplitudes
    data_gen_n0 = Epoch_Noise_Spiral_Generator(80, 20, 40, 10, 2, 0.05)
    data_gen_n1 = Epoch_Noise_Spiral_Generator(80, 20, 40, 10, 2, 0.075)
    data_gen_n2 = Epoch_Noise_Spiral_Generator(80, 20, 40, 10, 2, 0.1)
    data_gen_n3 = Epoch_Noise_Spiral_Generator(80, 20, 40, 10, 2, 0.25)
    data_gen_n4 = Epoch_Noise_Spiral_Generator(80, 20, 40, 10, 2, 0.5)

    data_gens = [data_gen_n0, data_gen_n1, data_gen_n2, data_gen_n3, data_gen_n4]
    noise_labels = ["5%", "7.5%", "10%", "25%", "50%"]
    colors = ["maroon", "goldenrod", "limegreen", "teal", "darkviolet"]
    device_param_labels = []

    models_ode_rnn = []
    models_gru_rnn = []

    output_ode_rnns_list = []
    cmap_ode_rnns_list = []

    output_gru_rnns_list = []
    cmap_gru_rnns_list = []

    total_loss_ode = []
    total_loss_gru = []

    model_loss_legend = []
    model_loss_legend.append(Line2D([0], [0], color="black", linestyle="solid", lw=2))
    model_loss_legend.append(Line2D([0], [0], color="black", linestyle="dashed", lw=2))

    train_loss_legend = []
    for color in colors:
        train_loss_legend.append(Line2D([0], [0], color=color, linestyle="solid", lw=2))

    device_params_list = []
    for i in range(0, 1):
        temp_device_params = copy.deepcopy(device_params)
        temp_device_params["viability"] = round(0.05 + 0.09 * i, 2)
        device_params_list.append(temp_device_params)
        device_param_labels.append(round(0.05 + 0.09 * i, 2))
    
    for device_param in device_params_list:

        # Plot loss history and average loss
        fig_loss, ax_loss = plt.subplots()

        output_ode_rnns = []
        output_gru_rnns = []
        cmap_ode_rnns = []
        cmap_gru_rnns = []

        for i in range(len(data_gens)):

            data_gen = data_gens[i]
            
            # Build, train, and plot model output
            ode_rnn = ODE_RNN_autogen(2, 6, 2, device_param, method, time_steps)
            losses_ode_rnn = ode_rnn_autogen_train(ode_rnn, data_gen, epochs)
            output_ode_rnn = ode_rnn_autogen_test(ode_rnn, data_gen)
            output_ode_rnns.append(output_ode_rnn)
            models_ode_rnn.append(ode_rnn)

            gru_rnn = GRU_RNN_autogen(2, 6, 2, device_param)
            losses_gru_rnn = gru_rnn_autogen_train(gru_rnn, data_gen, epochs)
            output_gru_rnn = gru_rnn_autogen_test(gru_rnn, data_gen)
            output_gru_rnns.append(output_gru_rnn)
            models_gru_rnn.append(gru_rnn)

            # Plot crossbar mapping and loss
            fig_cmap_ode, ax_cmap_ode = plot_cmap(ode_rnn)
            cmap_ode_rnns.append([fig_cmap_ode, ax_cmap_ode])
            fig_loss_ode, ax_loss_ode = plot_loss(epochs, losses_ode_rnn)

            fig_cmap_gru, ax_cmap_gru = plot_cmap(gru_rnn)
            cmap_gru_rnns.append([fig_cmap_gru, ax_cmap_gru])
            fig_loss_gru, ax_loss_gru = plot_loss(epochs, losses_gru_rnn)

            ax_loss.plot(list(range(epochs)), losses_ode_rnn, color=colors[i], linewidth=1, linestyle="solid")
            ax_loss.plot(list(range(epochs)), losses_gru_rnn, color=colors[i], linewidth=1, linestyle="dashed")

            # Append the loss to a list for future graphing
            total_loss_ode.append(losses_ode_rnn)
            total_loss_gru.append(losses_gru_rnn)

        output_ode_rnns_list.append(output_ode_rnns)
        output_gru_rnns_list.append(output_gru_rnns)
        cmap_ode_rnns_list.append(cmap_ode_rnns)
        cmap_gru_rnns_list.append(cmap_gru_rnns)

        # Plot all axis labels
        plt.xlabel("Epoch")
        plt.ylabel("MSE loss on +" + str(device_param["viability"]) + " variability crossbar")

        model_legend = ax_loss.legend(model_loss_legend, ["ODE-RNN", "GRU-RNN"], loc = "upper right")
        fig_loss.gca().add_artist(model_legend)
        ax_loss.legend(train_loss_legend, ["5%", "7.5%", "10%", "25%", "50%"], loc = (0.815, 0.545))
        fig_loss.savefig('./output/loss/' + str(device_param['viability']) + 'cmap_training_loss.png', dpi=600, transparent=True)

    for k in range(len(output_ode_rnns_list)):

        output_ode_rnns = output_ode_rnns_list[k]

        # Plot model outputs
        ode_rnn_fig, ode_rnn_axs = plt.subplots(nrows=2, ncols=3, figsize=(10, 12), subplot_kw=dict(projection='3d'))
        ode_rnn_axs[-1, -1].axis('off')
        gru_rnn_fig, gru_rnn_axs = plt.subplots(nrows=2, ncols=3, figsize=(10, 12), subplot_kw=dict(projection='3d'))
        gru_rnn_axs[-1, -1].axis('off')

        # Configure tight layout for 2x3 plots
        ode_rnn_fig.tight_layout()
        gru_rnn_fig.tight_layout()

        ode_rnn_fig.subplots_adjust(top=0.9, hspace=0.2)
        gru_rnn_fig.subplots_adjust(top=0.9, hspace=0.2)

        ode_rnn_fig.suptitle("ODE-RNN +" + str(device_param_labels[k]) + " crossbar variability", fontsize = 16)
        gru_rnn_fig.suptitle("GRU-RNN +" + str(device_param_labels[k]) + " crossbar variability", fontsize = 16)

        # Plot each of the 3D outputs of each ODE RNN model
        count = 0
        for i, output_ax_ode in enumerate(ode_rnn_axs.flat):

            if count == 5:
                break

            title_text = "Training noise +" + noise_labels[count]
            output_ax_ode.set_title(title_text)

            data_gen = data_gens[count]

            output_ax_ode.plot3D(data_gen.true_x, data_gen.true_y, data_gen.true_z, 'gray')
            d1, d2, d3 = data_gen.y[0, :].squeeze(), data_gen.y[1, :].squeeze(), data_gen.x.squeeze()
            output_ax_ode.scatter3D(d1, d2, d3, 'blue')

            output_ode_rnn = output_ode_rnns[count]
            output_ax_ode.plot3D(output_ode_rnn[0], output_ode_rnn[1], output_ode_rnn[2], color="black", linewidth=1.5)

            count += 1

        # Plot each of the 3D outputs of each GRU RNN model
        count = 0
        for i, output_ax_gru in enumerate(gru_rnn_axs.flat):

            if count == 5:
                break

            title_text = "Training noise +" + noise_labels[count]
            output_ax_gru.set_title(title_text)

            data_gen = data_gens[count]

            output_ax_gru.plot3D(data_gen.true_x, data_gen.true_y, data_gen.true_z, 'gray')
            d1, d2, d3 = data_gen.y[0, :].squeeze(), data_gen.y[1, :].squeeze(), data_gen.x.squeeze()
            output_ax_gru.scatter3D(d1, d2, d3, 'blue')

            output_gru_rnn = output_gru_rnns[count]
            output_ax_gru.plot3D(output_gru_rnn[0], output_gru_rnn[1], output_gru_rnn[2], color="black", linewidth=1.5)

            count += 1

        ode_rnn_fig.savefig('./output/output/' + str(device_param_labels[k]) + 'ode_noise_comp.png', dpi=600, transparent=True)
        gru_rnn_fig.savefig('./output/output/' + str(device_param_labels[k]) + 'gru_noise_comp.png', dpi=600, transparent=True)

    # Save all cmap figures
    for cmap_ode_rnns in cmap_ode_rnns_list:
        for i in range (len(cmap_ode_rnns)):
            cmap_ode = cmap_ode_rnns[i]
            save_name = "./output/cmap/ode_rnn" + str(i) + ".png"
            cmap_ode[0].savefig(save_name, dpi=600, transparent=True)

    for cmap_gru_rnns in cmap_gru_rnns_list:
        for i in range (len(cmap_gru_rnns)):
            cmap_gru = cmap_gru_rnns[i]
            save_name = "./output/cmap/gru_rnn" + str(i) + ".png"
            cmap_gru[0].savefig(save_name, dpi=600, transparent=True)

def single_model_plot_hard(epochs, device_params, method, time_steps):

    # Create data generators with different nosie amplitudes
    data_gen_n0 = Epoch_AM_Wave_Generator(80, 80, 2, 10, 2, 0.05)
    data_gen_n1 = Epoch_AM_Wave_Generator(80, 80, 2, 10, 2, 0.075)
    data_gen_n2 = Epoch_AM_Wave_Generator(80, 80, 2, 10, 2, 0.10)
    data_gen_n3 = Epoch_AM_Wave_Generator(80, 80, 2, 10, 2, 0.25)
    data_gen_n4 = Epoch_AM_Wave_Generator(80, 80, 2, 10, 2, 0.50)

    data_gens = [data_gen_n0, data_gen_n1, data_gen_n2, data_gen_n3, data_gen_n4]
    noise_labels = ["0.05", "0.075", "0.1", "0.25", "0.5"]
    colors = ["maroon", "goldenrod", "limegreen", "teal", "darkviolet"]

    models_ode_rnn = []
    models_gru_rnn = []

    # Plot model outputs
    ode_rnn_fig, ode_rnn_axs = plt.subplots(nrows=2, ncols=3, figsize=(12, 12), subplot_kw=dict(projection='3d'))
    ode_rnn_axs[-1, -1].axis('off')

    output_ode_rnns = []
    cmap_ode_rnns = []

    gru_rnn_fig, gru_rnn_axs = plt.subplots(nrows=2, ncols=3, figsize=(12, 12), subplot_kw=dict(projection='3d'))
    gru_rnn_axs[-1, -1].axis('off')

    output_gru_rnns = []
    cmap_gru_rnns = []

    # Plot loss history and average loss
    fig_loss, ax_loss = plt.subplots()
    fig_loss.suptitle('Average MSE Loss')

    for i in range(len(data_gens)):

        data_gen = data_gens[i]
        
        # Build, train, and plot model output
        ode_rnn = ODE_RNN_autogen(2, 6, 2, device_params, method, time_steps)
        # print(sum(j.numel() for j in list(ode_rnn.parameters())))

        losses_ode_rnn = ode_rnn_autogen_train(ode_rnn, data_gen, epochs)
        output_ode_rnn = ode_rnn_autogen_test(ode_rnn, data_gen)
        output_ode_rnns.append(output_ode_rnn)
        models_ode_rnn.append(ode_rnn)

        gru_rnn = GRU_RNN_autogen(2, 6, 2, device_params)
        # print(sum(j.numel() for j in list(gru_rnn.parameters())))

        losses_gru_rnn = gru_rnn_autogen_train(gru_rnn, data_gen, epochs)
        output_gru_rnn = gru_rnn_autogen_test(gru_rnn, data_gen)
        output_gru_rnns.append(output_gru_rnn)
        models_gru_rnn.append(gru_rnn)

        # Plot crossbar mapping and loss
        fig_cmap_ode, ax_cmap_ode = plot_cmap(ode_rnn)
        cmap_ode_rnns.append([fig_cmap_ode, ax_cmap_ode])
        fig_loss_ode, ax_loss_ode = plot_loss(epochs, losses_ode_rnn)

        fig_cmap_gru, ax_cmap_gru = plot_cmap(gru_rnn)
        cmap_gru_rnns.append([fig_cmap_gru, ax_cmap_gru])
        fig_loss_gru, ax_loss_gru = plot_loss(epochs, losses_gru_rnn)

        ax_loss.plot(list(range(epochs)), losses_ode_rnn, color=colors[i], linewidth=1, linestyle="solid")
        ax_loss.plot(list(range(epochs)), losses_gru_rnn, color=colors[i], linewidth=1, linestyle="dashed")

    # Plot each of the 3D outputs of each ODE RNN model
    count = 0
    for i, output_ax_ode in enumerate(ode_rnn_axs.flat):

        if count == 5:
            break

        title_text = "ODE-RNN noise +" + noise_labels[count]
        output_ax_ode.set_title(title_text)

        data_gen = data_gens[count]

        output_ax_ode.plot3D(data_gen.true_x.squeeze(), data_gen.true_y.squeeze(), data_gen.true_z.squeeze(), 'gray')
        d1, d2, d3 = data_gen.y[0, :].squeeze(), data_gen.y[1, :].squeeze(), data_gen.x.squeeze()
        output_ax_ode.scatter3D(d1, d2, d3, 'blue')

        output_ode_rnn = output_ode_rnns[count]
        output_ax_ode.plot3D(output_ode_rnn[0], output_ode_rnn[1], output_ode_rnn[2], color="black", linewidth=1.5)

        count += 1

    # Plot each of the 3D outputs of each GRU RNN model
    count = 0
    for i, output_ax_gru in enumerate(gru_rnn_axs.flat):

        if count == 5:
            break

        title_text = "GRU-RNN noise +" + noise_labels[count]
        output_ax_gru.set_title(title_text)

        data_gen = data_gens[count]

        output_ax_gru.plot3D(data_gen.true_x.squeeze(), data_gen.true_y.squeeze(), data_gen.true_z.squeeze(), 'gray')
        d1, d2, d3 = data_gen.y[0, :].squeeze(), data_gen.y[1, :].squeeze(), data_gen.x.squeeze()
        output_ax_gru.scatter3D(d1, d2, d3, 'blue')

        output_gru_rnn = output_gru_rnns[count]
        output_ax_gru.plot3D(output_gru_rnn[0], output_gru_rnn[1], output_gru_rnn[2], color="black", linewidth=1.5)

        count += 1

    all_loss = []
    all_loss.append(Line2D([0], [0], color="black", linestyle="solid", lw=4))
    all_loss.append(Line2D([0], [0], color="black", linestyle="dashed", lw=4))

    for color in colors:
        all_loss.append(Line2D([0], [0], color=color, linestyle="solid", lw=4))

    # Configure tight layout for 2x3 plots
    ode_rnn_fig.tight_layout()
    gru_rnn_fig.tight_layout()

    # Plot all axis labels
    ax_loss.legend(all_loss, ["ODE-RNN", "GRU-RNN", "0.05", "0.075", "0.1", "0.25", "0.5"])

    # Save all figures
    for i in range (len(cmap_ode_rnns)):

        cmap_ode = cmap_ode_rnns[i]

        save_name = "./output/cmap/ode_rnn" + str(i) + ".png"
        cmap_ode[0].savefig(save_name, dpi=600, transparent=True)

    for i in range (len(cmap_gru_rnns)):

        cmap_gru = cmap_gru_rnns[i]

        save_name = "./output/cmap/gru_rnn" + str(i) + ".png"
        cmap_gru[0].savefig(save_name, dpi=600, transparent=True)

    fig_loss.savefig('./output/model_training_difference.png', dpi=600, transparent=True)
    ode_rnn_fig.savefig('./output/ode_noise_comp.png', dpi=600, transparent=True)
    gru_rnn_fig.savefig('./output/gru_noise_comp.png', dpi=600, transparent=True)

# Device parameters for convenience
device_params = {"Vdd": 0.2,
                 "r_wl": 20,
                 "r_bl": 20,
                 "m": 72,
                 "n": 72,
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

# data_gen = Epoch_Noise_Spiral_Generator(80, 20, 40, 10, 2, 0.05)

single_model_plot(30, device_params, "euler", 1)
# single_model_plot_hard(125, device_params, "euler", 0.1)
# graph_step_size_difference(1, 30, "euler", device_params, data_gen)

# graph_average_performance(1, 30, device_params, "euler", 1)
# graph_ode_solver_difference(10, 30, device_params)
# graph_step_size_difference(1, 30, "rk4", device_params)

# graph_model_difference(1, 30, device_params, "euler", 1)

# data_gen = Epoch_AM_Wave_Generator(80, 80, 2, 10, 2, 0.00)

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