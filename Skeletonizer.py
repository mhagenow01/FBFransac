from collections import defaultdict
import numpy as np
import time
from pstats import Stats
from cProfile import Profile
from scipy.ndimage.morphology import distance_transform_edt
from scipy.ndimage import correlate
import scipy.ndimage.morphology
from Graph import Graph

# This bounding box is just a hard-coded region around the hand.
origin = np.array((-0.3,-0.2,0.2))
extent = np.array((0, 0, 0.5))

# These parameters are specific to the mesh.
binsize = 0.001
minDistance = 0.003
maxDistance = 0.006



def distanceFieldFromCloud(cloud):
    ''' Given a cloud, make 3d binary voxel grid
        Then get the distance to the nearest 1 voxel at each grid point.
    '''
    shape = np.array((extent - origin) // binsize, dtype = np.int)
    
    distanceField = np.full(shape, 1)
    for c in cloud:
        index = np.array((c - origin) // binsize, dtype=np.int)
        if np.all(index >= 0) and np.all(index < shape):
            distanceField[tuple(index)] = 0
    distanceField = distance_transform_edt(distanceField, binsize)
    return distanceField

def cloudFromMask(mask):
    ''' Given a binary voxel grid, recover a point cloud that would hash to it'''
    return ((np.array(np.where(mask)) + 0.5) * binsize).T + origin

def gradient(distanceField):
    ''' Compute the gradient of the distance field with a 
        Simple filter.

        This can take a while for large grids. And I am not sure how to 
        extend it to sparse ones.
    '''
    fx = correlate(distanceField, np.array([
        [[-1, 0, 1]]
    ]), mode = 'nearest') / (binsize * 2)
    fy = correlate(distanceField, np.array([
        [[-1],
        [0],
        [1]]
    ]), mode = 'nearest') / (binsize * 2)
    fz = correlate(distanceField, np.array([
        [[-1]], [[0]], [[1]]
    ]), mode = 'nearest') / (binsize * 2)
    return fx, fy, fz

def findSkeleton(distanceField):
    ''' The magnitude of the gradient vector when the 
        min function is not active is always 1. So we will
        look for points that have a magnitude of < 0.8 to see where the 
        min function is operating.
    '''
    fx, fy, fz = gradient(distanceField)
    return (fx**2 + fy**2 + fz**2 < 0.8) & (distanceField > minDistance) & (distanceField < maxDistance)

def graphIterate(skeleton, n = 10, minNodes = 500):
    g = Graph()
    g.fromMatrix(skeleton)
    g.prune(n, minNodes)
    skeleton = g.toMatrix(np.zeros_like(skeleton, dtype = np.bool))
    return skeleton

def skeletonizer(cloud):
    field = distanceFieldFromCloud(cloud)
    skeleton = findSkeleton(field)

    skg = graphIterate(skeleton)
    print(f'Final: {np.sum(skg)}')
    print(f'Distances: {field[skg]}')
    return skg
