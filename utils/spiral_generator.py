
import torch

import numpy as np
import matplotlib.pyplot as plt

from mpl_toolkits import mplot3d

class Regular_Spiral_Generator():

    def __init__(self, n_pts, cutoff, batch_time, batch_size):

        # Store instance variables
        self.n_pts = n_pts
        self.cutoff = cutoff
        self.batch_time = batch_time
        self.batch_size = batch_size

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

    def get_random_batch(self):
        s = torch.from_numpy(np.random.choice(np.arange(self.n_pts - self.batch_time, dtype=np.int64), self.batch_size, replace=False))
        batch_y0 = self.d[s]  # (M, D)
        batch_t = self.t[:self.batch_time]  # (T)
        batch_y = torch.stack([self.d[s + i] for i in range(self.batch_time)], dim=0)  # (T, M, D)
        return batch_y0.to(), batch_t.to(), batch_y.to()