import matplotlib.pyplot as plt 
import numpy as np 
import json
import functools
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from pykdtree.kdtree import KDTree
from scipy.spatial import cKDTree

class SupportSphere:
    MaxShift = 0.001
    MinShift = 0.00005
    dr = 0.00001
    Epsilon = 0.00000001
    MaxIterations = 1000
    def __init__(self, x):
        self.X = x
        self.R = 0
        self.Iterations = 1
    
    def update(self, cloud, kd : cKDTree):
        self.Iterations += 1
        if self.Iterations > self.MaxIterations:
            return False

        distance, index = kd.query(self.X, 4, distance_upper_bound = self.R + self.dr)
        distance, index = distance[0], index[0]
        index = index[~np.isinf(distance)]
        distance = distance[~np.isinf(distance)]
        if len(distance) > 3:
            return True
        if len(distance) == 0:
            self.R += self.dr
            return None
        if len(distance) == 1:
            p = cloud[index[0]]
            dx = self.X - p
            dx *= self.dr / np.linalg.norm(dx)
            self.X += dx
            self.R += self.dr
        if len(distance) == 2:
            p1, p2 = cloud[index]
            midPoint = (p1 + p2) / 2
            dx = self.X - midPoint
            dx *= np.sqrt((self.R + self.dr) ** 2 - self.R ** 2) / np.linalg.norm(dx)
            self.X += dx
            self.R += self.dr
        if len(distance) == 3:
            p1, p2, p3 = cloud[index]
            n = np.cross(p2 - p1, p3 - p1)
            self.X += self.dr * n / np.linalg.norm(n) / 4
        return None

        
def drawSphere(ax, X, R):
    u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
    x = X[0,0] + R * np.cos(u)*np.sin(v)
    y = X[0,1] + R * np.sin(u)*np.sin(v)
    z = X[0,2] + R * np.cos(v)
    ax.plot_surface(x, y, z, color="r")

r = 0.008
spheres = [SupportSphere(np.array([(0.0,0.0,0.0)]))]
def update(ax, cloudArtist, cloud, kd, t):
    global spheres
    for i in range(50):
        s = spheres[-1]
        converged = s.update(cloud, kd)
        if converged is not None or s.R > r:
            bottomLeft = np.min(cloud, axis = 0).reshape((1,3))
            topRight = np.max(cloud, axis = 0).reshape((1,3))
            nextStart = np.random.random(3) * (topRight - bottomLeft) + bottomLeft
            if converged:
                spheres.append(
                    SupportSphere(
                        nextStart
                    )
                )
            else:
                spheres[-1] = SupportSphere(
                                nextStart
                            )
    ax.clear()
    cloudArtist = ax.scatter(cloud[:,0], cloud[:,1], cloud[:,2])
    for s in spheres:
        drawSphere(ax, s.X, s.R)
    return None

if __name__ == '__main__':
    with open('Models/Cloud_ToyScrew-Yellow.json') as fin:
        cloud = np.array(json.load(fin))
    
    ax = plt.gca(projection = '3d')
    cloudArtist = ax.scatter(cloud[:,0], cloud[:,1], cloud[:,2])

    ani = FuncAnimation(plt.gcf(), functools.partial(update, ax, cloudArtist, cloud, cKDTree(cloud)), [1])

    plt.show()

    