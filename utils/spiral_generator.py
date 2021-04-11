
import torch

import numpy as np
import matplotlib.pyplot as plt

from mpl_toolkits import mplot3d

class Regular_Spiral_Generator():

    def __init__(self, n_pts, cutoff):

        # z doubles as time

        # Data for a three-dimensional line
        self.z = torch.linspace(0, 20, n_pts)
        self.x = torch.sin(self.z)
        self.y = torch.cos(self.z)

        # Data for three-dimensional irregular time sampled observations
        self.z_obs = 20 * torch.rand(100)
        self.x_obs = torch.sin(self.z_obs) + 0.1 * np.random.randn(100)
        self.y_obs = torch.cos(self.z_obs) + 0.1 * np.random.randn(100)

        self.t = self.z
        self.d = torch.cat((self.x, self.y), axis=0)

        self.train = self.d[:cutoff]
        self.validation = self.d[cutoff:]

    def plot(self):

        ax = plt.axes(projection='3d')
        ax.plot3D(self.x, self.y, self.z, 'gray')
        ax.scatter3D(self.x_obs, self.y_obs, self.z_obs, c=self.z_obs, cmap='Blues')

        plt.show()

    def get_plot(self):
        
        ax = plt.axes(projection='3d')
        ax.plot3D(self.x, self.y, self.z, 'gray')
        ax.scatter3D(self.x_obs, self.y_obs, self.z_obs, c=self.z_obs, cmap='Blues')

        return ax


