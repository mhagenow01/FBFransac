import matplotlib.pyplot as plt 
import numpy as np 
import json
import functools
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from pykdtree.kdtree import KDTree
from scipy.spatial import cKDTree

EPSILON = 0.00001

class BetterSupportSphere:
    def __init__(self, x, maxIter = 100):
        self.X = x
        self.Iterations = 0
        self.MaxIter = maxIter
    
    def update(self, cloud, kd, radius, dr):

        self.Iterations += 1
        if self.Iterations >= self.MaxIter:
            return False

        distance, index = kd.query(self.X, 100, distance_upper_bound = 1.1*radius)
        distance, index = distance[0], index[0]
        index = index[~np.isinf(distance)]
        
        distance = distance[~np.isinf(distance)]
        if len(distance) < 20:
            return False
        internalPoints = distance < (radius - dr/2)
        if ~np.any(internalPoints) and sum(distance <= radius + dr/2) > 20:
            return True
        
        points = cloud[index]
        dx = self.X - points
        dx = ((dx.T) / np.linalg.norm(dx, axis = 1)).T
        
        weights = 1 / (1 + np.exp((distance - radius) / (dr/2)))
        #weights = np.exp(-(distance)**2 / (2 * radius**2))
        dx = np.sum((((radius - distance) * dx.T) * weights) / np.sum(weights), axis = 1).T
        self.X += dx
        if np.linalg.norm(dx) < EPSILON:
            return False
        return None
   
def drawSphere(ax, X, R):
    u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
    x = X[0,0] + R * np.cos(u)*np.sin(v)
    y = X[0,1] + R * np.sin(u)*np.sin(v)

    z = X[0,2] + R * np.cos(v)
    ax.plot_surface(x, y, z, color="r")

r = 0.0064
dr = 0.001
spheres = [BetterSupportSphere(np.array([[0.,0.,0.]]))]
def update(ax, cloud, kd, t):
    global spheres
    pos = None
    while pos is None:
        pos = spheres[-1].update(cloud, kd, r, dr)
    bottomLeft = np.min(cloud, axis = 0).reshape((1,3))
    topRight = np.max(cloud, axis = 0).reshape((1,3))
    nextPos = np.random.random(3) * (topRight - bottomLeft) + bottomLeft
    if isinstance(pos, bool):
        spheres[-1] = BetterSupportSphere(np.array(nextPos))
    else:
        spheres.append(BetterSupportSphere(np.array(nextPos)))
    ax.clear()
    ax.scatter(cloud[:,0], cloud[:,1], cloud[:,2])
    for s in spheres:
        drawSphere(ax, s.X, r)
    return None


def profileObject(cloud, minR, maxR, stepSize):
    kd = cKDTree(cloud)
    keyPoints = {}
    bottomLeft = np.min(cloud, axis = 0).reshape((1,3))
    topRight = np.max(cloud, axis = 0).reshape((1,3))
    for r in np.arange(minR, maxR+0.00000001, stepSize):
        spheres = []
        for i in range(1000):
            startPos = np.random.random(3) * (topRight - bottomLeft) + bottomLeft
            s = BetterSupportSphere(startPos)
            found = None
            while found is None:
                found = s.update(cloud, kd, r, 0.1 * r)
            if found:
                spheres.append(s)
        keyPoints.append((r, spheres))
    return keyPoints




if __name__ == '__main__':
    with open('Models/Cloud_ToyScrew-Yellow.json') as fin:
        cloud = np.array(json.load(fin))
        #cloud = cloud[cloud[:,0] > 0]
    
    keyPoints = profileObject(cloud, 0.003, 0.008, 0.0002)
    for k,v in keyPoints.items():
        print(k, len(v))

    