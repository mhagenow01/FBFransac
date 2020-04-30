from test_sphere_thing import ObjectProfile, SupportSphere
import pickle
import sys
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np


def drawSphere(ax, pos, r):
    u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
    x = r * np.cos(u)*np.sin(v) + pos[0,0]
    y = r * np.sin(u)*np.sin(v) + pos[0,1]
    z = r * np.cos(v) + pos[0,2]
    ax.plot_wireframe(x, y, z, color="r")

if __name__ == '__main__':
    file = sys.argv[1]
    with open(file, 'rb') as fin:
        profile = pickle.load(fin)
    
    ax = plt.gca(projection = '3d')
    for r, spheres in profile.KeyPoints:
        for s in spheres:
            drawSphere(ax, s.X, r)

    plt.show()
    