
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

from utils.spiral_generator import Regular_Spiral_Generator

from networks.ode import ODE_Func as ODE_Net
from networks.ode import train as ode_train

from networks.lstm_rnn import LSTM_RNN as LSTM_RNN
from networks.lstm_rnn import train as lstm_train

# Device parameters for convenience        
device_params = {"Vdd": 0.2,
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
                 "r_on_stddev": 1e3,
                 "r_off_stddev": 1e4,
                 "p_stuck_on": 0.01,
                 "p_stuck_off": 0.01,
                 "method": "viability",
                 "viability": 0.05,
}

# Get regular spiral data with irregularly sampled time intervals (+ noise)
data_gen = Regular_Spiral_Generator(50, 50, 25, 10, 20)
ground_truth = data_gen.get_plot()
plt.savefig('./output/ground_truth.png', dpi=600, transparent=True)

#data_gen.plot_random_batch()

# Build and train models

# ode_net = ODE_Net(3, 50, 3, crossbar(device_params))
# ode_train(ode_net, data_gen, 500)

lstm_rnn = LSTM_RNN(3, 50, 3, device_params)
lstm_train(lstm_rnn, data_gen, 2000)

# Test models


# Display all remaining plots
plt.show()
