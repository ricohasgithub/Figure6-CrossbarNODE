
import torch
import torch.nn as nn
import torch.optim as optim
import torch.functional as F

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from mpl_toolkits import mplot3d

from utils.spiral_generator import Regular_Spiral_Generator

data_gen = Regular_Spiral_Generator()
data_gen.plot()