import matplotlib.pyplot as plt 
import numpy as np 
import json
import functools
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from pykdtree.kdtree import KDTree
from scipy.spatial import cKDTree

class SupportSphere:
    dr = 0.01
    Epsilon = 0.0001
    MaxIterations = 100
    def __init__(self, x):
        self.X = x
        self.R = 0
        self.Iterations = 1
    
    def update(self, cloud, kd : cKDTree):
        self.Iterations += 1
        if self.Iterations > self.MaxIterations:
            print('ahh')
            return False

        distance, index = kd.query(self.X, 400, eps=0, distance_upper_bound = self.R + self.dr)
        distance, index = distance[0], index[0]
        index = index[~np.isinf(distance)]
        distance = distance[~np.isinf(distance)]
        touchingDistance = distance[distance <= self.R + self.Epsilon]
        touchingIndex = index[distance <= self.R + self.Epsilon]
        if len(touchingDistance) >= 3:
            return True
        if len(distance) == 0:
            self.R += self.dr
            return None

        if len(touchingDistance) == 0:
            dr = distance[0] - self.R
            self.R += dr
        if len(touchingDistance) == 1:
            p = cloud[touchingIndex[0]]
            dr = self.dr
            dx = self.X - p
            dx /= np.linalg.norm(dx)
            dr = self.tryMove(dx, dr, cloud, index[1:])
        if len(touchingDistance) == 2:
            p1, p2 = cloud[touchingIndex]
            midPoint = (p1 + p2) / 2
            dr = self.dr
            dx = self.X - midPoint
            dx /= np.linalg.norm(dx)
            a = np.linalg.norm(p1 - midPoint)
            b = np.sqrt((self.R + self.dr)**2 - a**2)
            dr = self.tryMove(dx, b - np.sqrt(self.R**2 - a**2), cloud, index[2:])
        if len(touchingDistance) == 3:
            p1, p2, p3 = cloud[touchingIndex]
            n = np.cross(p2 - p1, p3 - p1)
            dr = self.dr / 4
            self.X += dr * n / np.linalg.norm(n)
        if dr < self.Epsilon:
            return True
        return None

    def tryMove(self, dx, d, cloud, nearByPointsIndex):
        nearbyPoints = cloud[nearByPointsIndex]
        distanceAlongAxis = np.sum(dx * (nearbyPoints - self.X), axis = 1)
        distanceToAxis = np.linalg.norm(nearbyPoints - self.X - np.outer(distanceAlongAxis, dx), axis = 1)

        distanceAlongAxis = distanceAlongAxis[distanceToAxis <= self.R]
        distanceToAxis = distanceToAxis[distanceToAxis <= self.R]
        
        distanceToCollision = (np.abs(distanceAlongAxis) - np.sqrt(self.R**2 - distanceToAxis**2)) \
                                * np.sign(distanceAlongAxis)
        willCollide = (distanceToCollision > 0) & (distanceToCollision < 2 * d)
        prevDistance = np.linalg.norm(nearbyPoints - self.X, axis = 1)

        if np.any(willCollide):
            d = np.min(distanceToCollision[willCollide]) / 2
        self.X += dx * d
        return d
        
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
    for i in range(1):
        s = spheres[-1]
        converged = s.update(cloud, kd)
        if converged is not None or s.R > r:
            bottomLeft = np.min(cloud, axis = 0).reshape((1,3))
            topRight = np.max(cloud, axis = 0).reshape((1,3))
            nextStart = np.random.random(3) * (topRight - bottomLeft) + bottomLeft
            if converged:
                spheres.append(SupportSphere(nextStart))
            else:
                spheres[-1] = SupportSphere(nextStart)
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

    