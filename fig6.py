
import torch
import torch.nn as nn
import torch.optim as optim
import torch.functional as F

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from mpl_toolkits import mplot3d

from utils.spiral_generator import Regular_Spiral_Generator

# Device parameters for convenience        
device_params = {"Vdd": 0.2,
                 "r_wl": 20,
                 "r_bl": 20,
                 "m": 32,
                 "n": 32,
                 "r_on_mean": 1e4,
                 "r_on_stddev": 1e3,
                 "r_off_mean": 1e5,
                 "r_off_stddev": 1e4,
                 "dac_resolution": 4,
                 "adc_resolution": 14,
                 "device_resolution": 4,
                 "bias_scheme": 1/3,
                 "tile_rows": 8,
                 "tile_cols": 8,
                 "r_cmos_line": 600,
                 "r_cmos_transistor": 20, 
                 "p_stuck_on": 0.001,
                 "p_stuck_off": 0.001
            }

# Get regular spiral data with irregularly sampled time intervals (+ noise)
data_gen = Regular_Spiral_Generator(50, 1000, 500, 10, 20)
ground_truth = data_gen.get_plot()
plt.savefig('./output/ground_truth.png', dpi=600, transparent=True)

# Build models

# Display all remaining plots
plt.show()
