
import torch

import random
import numpy as np
import matplotlib.pyplot as plt

from mpl_toolkits import mplot3d

class Epoch_Spiral_Generator():

    def __init__(self, n_pts, cutoff, depth, train_window, dimension, seq_length):

        # Store instance variables
        self.n_pts = n_pts
        self.cutoff = cutoff
        self.depth = depth
        self.train_window = train_window
        self.dimension = dimension

        # Generate spiral
        self.x = torch.linspace(0, depth, n_pts).reshape(1, -1) + 1

        self.y_x = (torch.sin(self.x) + 0.05 * np.random.randn(n_pts)).float()
        self.y_y = (torch.cos(self.x) + 0.05 * np.random.randn(n_pts)).float()

        self.y = torch.cat((self.y_x, self.y_y), axis=0)
        
        self.true_z = torch.linspace(0, depth, n_pts).float() + 1
        self.true_x = torch.sin(self.true_z).float()
        self.true_y = torch.cos(self.true_z).float()
        self.true_data = torch.cat((self.true_x, self.true_y), axis=0)
        
        self.data = [((self.y[:, i:i+train_window].reshape(-1, dimension, 1), self.x[:, i:i+train_window].reshape(-1, 1, 1)), (self.y[:, i+train_window:i+train_window+train_window].reshape(dimension, -1))) for i in range(self.y.size(1) - train_window)]
        
        self.train_data = self.data[:cutoff]
        self.test_start = self.data[cutoff]

        self.test_x = (torch.cos(self.x) + 0.05 * np.random.randn(n_pts)).float()
        self.test_y = (torch.cos(self.x) + 0.05 * np.random.randn(n_pts)).float()
        self.true_data = torch.cat((self.test_x, self.test_y), axis=0)

        # self.test_data = [((self.y[:, i:i+seq_length].reshape(-1, dimension, 1), self.x[:, i:i+seq_length].reshape(-1, 1, 1)), (self.y[:, i+seq_length:(i+seq_length+1)].reshape(dimension, -1))) for i in range(self.y.size(1) - seq_length)]
        self.test_data = [((self.true_data[:, i:i+seq_length].reshape(-1, dimension, 1), self.x[:, i:i+seq_length].reshape(-1, 1, 1)), (self.true_data[:, i+seq_length:(i+seq_length+1)].reshape(dimension, -1))) for i in range(self.true_data.size(1) - seq_length)]
        # self.test_data = [(self.true_data[:, 0:80].reshape(-1, dimension, 1), self.x[:, 0:80].reshape(-1, 1, 1))]

class Epoch_Test_Spiral_Generator():

    def __init__(self, n_pts, cutoff, depth, train_window, dimension):

        # Store instance variables
        self.n_pts = n_pts
        self.cutoff = cutoff
        self.depth = depth
        self.train_window = train_window
        self.dimension = dimension

        self.x = torch.linspace(0, depth, n_pts).reshape(1, -1)

        self.y_x = (torch.cos(self.x) + 0.05 * np.random.randn(n_pts)).float()
        self.y_y = (torch.sin(self.x) + 0.05 * np.random.randn(n_pts)).float()

        self.y = torch.cat((self.y_x, self.y_y), axis=0)
        
        self.true_z = torch.linspace(0, depth, n_pts).float()
        self.true_x = torch.cos(self.true_z).float()
        self.true_y = torch.sin(self.true_z).float()

        self.data = [((self.y[:, i:i+train_window].reshape(-1, dimension, 1), self.x[:, i:i+train_window].reshape(-1, 1, 1)), (self.y[:, i+train_window:i+train_window+1].reshape(dimension, -1))) for i in range(self.y.size(1) - train_window)]

        # self.x = torch.linspace(0, depth, n_pts).reshape(1, -1)
        # self.y = torch.cat((torch.cos(self.x), torch.sin(self.x)), axis=0)
        # self.data = [((self.y[:, i:i+train_window].reshape(-1, dimension, 1), self.x[:, i:i+train_window].reshape(-1, 1, 1)), (self.y[:, i+train_window:i+train_window+1].reshape(dimension, -1))) for i in range(self.y.size(1) - train_window)]

        self.train_data = self.data[:cutoff]
        self.test_start = self.data[0:]

class Epoch_Noise_Spiral_Generator():

    def __init__(self, n_pts, cutoff, depth, train_window, dimension, noise):

        # Store instance variables
        self.n_pts = n_pts
        self.cutoff = cutoff
        self.depth = depth
        self.train_window = train_window
        self.dimension = dimension
        self.noise = noise

        self.x = torch.linspace(0, depth, n_pts).reshape(1, -1)

        self.y_x = (torch.cos(self.x) + noise * np.random.randn(n_pts)).float()
        self.y_y = (torch.sin(self.x) + noise * np.random.randn(n_pts)).float()

        self.y = torch.cat((self.y_x, self.y_y), axis=0)
        
        self.true_z = torch.linspace(0, depth, n_pts).float()
        self.true_x = torch.cos(self.true_z).float()
        self.true_y = torch.sin(self.true_z).float()

        self.data = [((self.y[:, i:i+train_window].reshape(-1, dimension, 1), self.x[:, i:i+train_window].reshape(-1, 1, 1)), (self.y[:, i+train_window:i+train_window+1].reshape(dimension, -1))) for i in range(self.y.size(1) - train_window)]

        # self.x = torch.linspace(0, depth, n_pts).reshape(1, -1)
        # self.y = torch.cat((torch.cos(self.x), torch.sin(self.x)), axis=0)
        # self.data = [((self.y[:, i:i+train_window].reshape(-1, dimension, 1), self.x[:, i:i+train_window].reshape(-1, 1, 1)), (self.y[:, i+train_window:i+train_window+1].reshape(dimension, -1))) for i in range(self.y.size(1) - train_window)]

        self.train_data = self.data[:cutoff]
        self.test_start = self.data[0:]

class Epoch_AM_Wave_Generator():

    def __init__(self, n_pts, cutoff, depth, train_window, dimension, noise):

        # Store instance variables
        self.n_pts = n_pts
        self.cutoff = cutoff
        self.depth = depth
        self.train_window = train_window
        self.dimension = dimension

        theta = torch.linspace(-4 * np.pi, 4 * np.pi, n_pts)

        self.x = torch.linspace(-2, 2, n_pts).reshape(1, -1)
        r = self.x ** 2 + 1
        self.y_x = r * (torch.sin(theta) + noise * np.random.randn(n_pts)).float()
        self.y_y = r * (torch.cos(theta) + noise * np.random.randn(n_pts)).float()

        self.x = self.x + 2
        self.y = torch.cat((self.y_x, self.y_y), axis=0)

        self.true_z = torch.linspace(-2, 2, n_pts).reshape(1, -1).float()
        r = self.true_z ** 2 + 1
        self.true_x = (r * torch.sin(theta)).float()
        self.true_y = (r * torch.cos(theta)).float()

        self.true_z = self.true_z + 2

        self.data = [((self.y[:, i:i+train_window].reshape(-1, dimension, 1), self.x[:, i:i+train_window].reshape(-1, 1, 1)), (self.y[:, i+train_window:i+train_window+1].reshape(dimension, -1))) for i in range(self.y.size(1) - train_window)]

        self.train_data = self.data[:cutoff]
        self.test_start = self.data[0:]

class Epoch_Spiral_Generator():
    
    def __init__(self, n_pts, cutoff, depth, train_window, dimension):

        # Store instance variables
        self.n_pts = n_pts
        self.cutoff = cutoff
        self.depth = depth
        self.train_window = train_window
        self.dimension = dimension

        self.x = torch.linspace(0, depth, n_pts).reshape(1, -1)

        self.a = 0.5
        self.b = 0.1
        #self.th = torch.linspace(475, 500, 10000).reshape(1, -1)
        self.y_x = self.a * torch.exp(self.b * self.x) * torch.cos(self.x)
        self.y_y = self.a * torch.exp(self.b * self.x) * torch.sin(self.x)
        # self.z = torch.linspace(0, 2, len(self.x))
        # self.x = torch.linspace(0, 2, len(self.th))
        
        # self.y_x = (torch.cos(self.x)).float()
        # self.y_y = (torch.sin(self.x) + torch.sin(self.x_2)).float()
        self.y = torch.cat((self.y_x, self.y_y), axis=0)

        self.data = [((self.y[:, i:i+train_window].reshape(-1, dimension, 1), self.x[:, i:i+train_window].reshape(-1, 1, 1)), (self.y[:, i+train_window:i+train_window+1].reshape(dimension, -1))) for i in range(self.y.size(1) - train_window)]

        self.train_data = self.data[:cutoff]
        self.test_start = self.data[0:]

        self.true_z = (torch.linspace(0, depth, n_pts).reshape(1, -1)).squeeze().float()

        self.true_x = (self.a * torch.exp(self.b * self.x) * torch.cos(self.true_z)).squeeze().float()
        self.true_y = (self.a * torch.exp(self.b * self.x) * torch.sin(self.true_z)).squeeze().float()

class Epoch_Square_Generator():
    
    def __init__(self, n_pts, cutoff, depth, train_window, dimension):

        # Store instance variables
        self.n_pts = n_pts
        self.cutoff = cutoff
        self.depth = depth
        self.train_window = train_window
        self.dimension = dimension

        self.x = torch.linspace(0, depth, n_pts).reshape(1, -1)

        self.y_x = (torch.cos(self.x) + 0.05 * np.random.randn(n_pts)).float()
        self.y_y = (torch.sin(self.x) + 0.05 * np.random.randn(n_pts)).float()

        self.y = torch.cat((self.y_x, self.y_y), axis=0)
        
        self.true_z = torch.linspace(0, depth, n_pts).float()
        self.true_x = torch.cos(self.true_z).float()
        self.true_y = torch.sin(self.true_z).float()

        self.data = [((self.y[:, i:i+train_window].reshape(-1, dimension, 1), self.x[:, i:i+train_window].reshape(-1, 1, 1)), (self.y[:, i+train_window:i+train_window+1].reshape(dimension, -1))) for i in range(self.y.size(1) - train_window)]

        # self.x = torch.linspace(0, depth, n_pts).reshape(1, -1)
        # self.y = torch.cat((torch.cos(self.x), torch.sin(self.x)), axis=0)
        # self.data = [((self.y[:, i:i+train_window].reshape(-1, dimension, 1), self.x[:, i:i+train_window].reshape(-1, 1, 1)), (self.y[:, i+train_window:i+train_window+1].reshape(dimension, -1))) for i in range(self.y.size(1) - train_window)]

        self.train_data = self.data[:cutoff]
        self.test_start = self.data[0:]

class Epoch_Heart_Generator():

    def __init__(self, n_pts, cutoff, depth, train_window, dimension, noise):

        # Store instance variables
        self.n_pts = n_pts
        self.cutoff = cutoff
        self.depth = depth
        self.train_window = train_window
        self.dimension = dimension

        self.x = torch.linspace(0, depth, n_pts).reshape(1, -1)
        self.x_2 = torch.linspace(0, 2*depth, n_pts).reshape(1, -1)

        self.y_x = (torch.cos(self.x_2) + noise * np.random.randn(n_pts)).float()
        self.y_y = (torch.sin(self.x) + torch.sin(self.x_2) + noise * np.random.randn(n_pts)).float()
        self.y = torch.cat((self.y_x, self.y_y), axis=0)

        self.data = [((self.y[:, i:i+train_window].reshape(-1, dimension, 1), self.x[:, i:i+train_window].reshape(-1, 1, 1)), (self.y[:, i+train_window:i+train_window+1].reshape(dimension, -1))) for i in range(self.y.size(1) - train_window)]

        self.train_data = self.data[:cutoff]
        self.test_start = self.data[0:]

        self.true_z = torch.linspace(0, depth, n_pts).squeeze().float()
        self.true_z_2 = torch.linspace(0, 2*depth, n_pts).squeeze().float()

        self.true_x = (torch.cos(self.true_z_2)).squeeze().float()
        self.true_y = (torch.sin(self.x) + torch.sin(self.true_z_2)).squeeze().float()

class Stochastic_Spiral_Generator():

    def __init__(self, n_pts, cutoff, depth, train_window, dimension):

        # Store instance variables
        self.n_pts = n_pts
        self.cutoff = cutoff
        self.depth = depth
        self.train_window = train_window
        self.dimension = dimension

        # Generate spiral
        self.x = torch.linspace(0, depth, n_pts).reshape(1, -1)

        self.y_x = (torch.sin(self.x) + 0.2 * np.random.randn(n_pts)).float()
        self.y_y = (torch.cos(self.x) + 0.2 * np.random.randn(n_pts)).float()

        self.theta = torch.linspace(0, 1.0, n_pts).reshape(1, -1)

        self.y = torch.mv(torch.cat((self.y_x, self.y_y), axis=0), self.theta)

        self.data = [((self.y[:, i:i+train_window].reshape(-1, dimension, 1), self.x[:, i:i+train_window].reshape(-1, 1, 1)), (self.y[:, i+train_window:i+train_window+10].reshape(dimension, -1))) for i in range(self.y.size(1) - train_window)]

class Regular_Spiral_Generator():

    def __init__(self, n_pts, cutoff, depth, batch_time, batch_size):

        # Store instance variables
        self.n_pts = n_pts
        self.cutoff = cutoff
        self.depth = depth
        self.batch_time = batch_time
        self.batch_size = batch_size

        # z doubles as time

        # Data for a three-dimensional line
        self.z = torch.linspace(0, depth, n_pts).float()
        self.x = torch.sin(self.z).float() * depth
        self.y = torch.cos(self.z).float() * depth

        # Data for three-dimensional irregular time sampled observations (+ noise)
        self.z_obs = (depth * torch.rand(n_pts)).float()
        self.x_obs = (torch.sin(self.z_obs) + 0.05 * np.random.randn(n_pts)).float() * depth
        self.y_obs = (torch.cos(self.z_obs) + 0.05 * np.random.randn(n_pts)).float() * depth

        self.t = self.z.float()
        self.obs = torch.stack((self.x_obs, self.y_obs, self.z_obs), dim=1).float()

        self.true_y = torch.stack((self.x, self.y, self.z), dim=1).float()
        self.true_y0 = self.true_y[0]

        #self.s = torch.arange(0, batch_time, step=1)

    def plot(self):

        ax = plt.axes(projection='3d')
        ax.plot3D(self.x, self.y, self.z, 'gray')
        ax.scatter3D(self.x_obs, self.y_obs, self.z_obs, cmap='Blues')

        plt.show()

    def get_plot(self):
        
        ax = plt.axes(projection='3d')
        ax.plot3D(self.x, self.y, self.z, 'gray')
        ax.scatter3D(self.x_obs, self.y_obs, self.z_obs, cmap='Blues')

        return ax

    def get_random_batch(self):

        s = torch.from_numpy(np.random.choice(np.arange(self.n_pts - self.batch_time, dtype=np.int64), self.batch_size, replace=False))
        batch_y0 = self.obs[s]  # (M, D)
        batch_t = self.t[:self.batch_time]  # (T)
        batch_y = torch.stack([self.obs[s + i] for i in range(self.batch_time)], dim=0)  # (T, M, D)
        return batch_y0.to(), batch_t.to(), batch_y.to()

    def plot_random_batch(self):

        batch_y0, batch_t, batch_y = self.get_random_batch()
        
        ax = plt.axes(projection='3d')
        ax.scatter3D(batch_y[0, :, 0], batch_y[0, :, 1], batch_y[0, : 2])
        ax.scatter3D(self.x_obs, self.y_obs, self.z_obs, cmap='Blues')

        return ax

    def plot_prediction(self, true_y, pred_y):

        ax = plt.axes(projection='3d')
        ax.plot3D(self.x, self.y, self.z, 'gray')
        ax.scatter3D(self.x_obs, self.y_obs, self.z_obs, cmap='Blues')
        ax.plot3D(pred_y[:, 0], pred_y[:, 1], pred_y[:, 2], 'red')

        plt.savefig('./output/ode.png', dpi=600, transparent=True)
        plt.show()