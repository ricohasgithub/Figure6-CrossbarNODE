
import numpy as np
import matplotlib.pyplot as plt

from mpl_toolkits import mplot3d

class Regular_Spiral_Generator():

    def __init__(self):

        # Data for a three-dimensional line
        self.z = np.linspace(0, 15, 1000)
        self.x = np.sin(self.z)
        self.y = np.cos(self.z)

        # Data for three-dimensional irregular time sampled observations
        self.z_obs = 15 * np.random.random(100)
        self.x_obs = np.sin(self.z_obs) + 0.1 * np.random.randn(100)
        self.y_obs = np.cos(self.z_obs) + 0.1 * np.random.randn(100)

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


