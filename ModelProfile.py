import matplotlib.pyplot as plt 
import numpy as np 
import json
import functools
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from pykdtree.kdtree import KDTree
from scipy.spatial import cKDTree
import pickle
import sys
import os
import glob
from PointCloudFromMesh import surfaceArea, pointCloudFromMesh

EPSILON = 0.00001
DENSITY = 50000
class SupportSphere:
    def __init__(self, x, maxIter = 100):
        self.X = x
        self.Iterations = 0
        self.MaxIter = maxIter

    def update(self, cloud, kd, radius):
        self.Iterations += 1
        if self.Iterations >= self.MaxIter:
            return False
        dr = radius * 0.01
        distance, index = kd.query(self.X, 10000, distance_upper_bound = radius + 2 * dr)
        distance, index = distance[0], index[0]
        index = index[~np.isinf(distance)]
        
        distance = distance[~np.isinf(distance)]
        area = 4 * np.pi * radius**2
        if len(distance) <= 3:
            return False
        if len(distance[distance < radius + dr/2])/(DENSITY * area) < 0.15:
            return False
        internalPoints = distance < (radius - dr/2)
        weights = 1 / (1 + np.exp((distance - radius) / dr))
        mean = np.average(distance, weights= weights)
        if abs(mean - radius) <= dr / 2:
            print(mean, mean - radius)
            return True
        
        points = cloud[index]
        dx = self.X - points
        dx = ((dx.T) / np.linalg.norm(dx, axis = 1)).T
        
        dx = np.sum((((radius - distance) * dx.T) * weights) / np.sum(weights), axis = 1).T
        self.X += dx
        if np.linalg.norm(dx) < EPSILON:
            return False
        return None

    def iterate(self, cloud, kd, r):
        found = None
        while found is None:
            found = self.update(cloud, kd, r)
        return found

class ObjectProfile:
    def __init__(self, meshFile):
        area, scale = surfaceArea(meshFile)
        cloud = pointCloudFromMesh(meshFile, int(DENSITY * area))
        print(cloud.shape)
        radii = np.arange(0.05, 0.25, 0.001) * np.min(scale)
        self.findKeyPoints(radii, cloud)
    
    def findKeyPoints(self, radii, cloud):
        kd = cKDTree(cloud)
        self.KeyPoints = []
        bottomLeft = np.min(cloud, axis = 0).reshape((1,3))
        topRight = np.max(cloud, axis = 0).reshape((1,3))
        for r in radii:
            spheres = []
            for i in range(1000):
                startPos = np.random.random(3) * (topRight - bottomLeft) + bottomLeft
                s = SupportSphere(startPos)
                found = None
                while found is None:
                    found = s.update(cloud, kd, r)
                if found:
                    spheres.append(s)
            print(r, len(spheres))
            if len(spheres) > 0:
                self.KeyPoints.append((r, spheres))
        
        for k,v in self.KeyPoints:
            print(k, len(v))
        return
    
    def sampleRadius(self):
        p = np.array([len(v) for r,v in self.KeyPoints])
        p = p / np.sum(p)
        return self.KeyPoints[np.random.choice(range(len(self.KeyPoints)),p = p)]

    @staticmethod
    def fromMeshFile(file):
        path, name = os.path.split(file)
        with open(os.path.join('ObjectProfiles', name), 'rb') as fin:
            return pickle.load(fin)

if __name__ == '__main__':
    globString = sys.argv[1]

    for file in glob.glob(globString):
        print(f'Profiling {file}')
        with open(file) as fin:
            profile = ObjectProfile(file)

        directory, name = os.path.split(file)
        with open(f'ObjectProfiles\\{name}', 'wb') as fout:
            pickle.dump(profile, fout)

    