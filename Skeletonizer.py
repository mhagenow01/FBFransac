from collections import defaultdict
import numpy as np
import time
from pstats import Stats
from cProfile import Profile
from scipy.ndimage.morphology import distance_transform_edt
from scipy.ndimage import correlate

origin = np.array((-0.03, -0.03, -0.03))
extent = np.array((0.03, 0.03, 0.03))
binsize = 0.001
theta = np.pi / 4
threshold =  np.sqrt(2 - 2 * np.cos(theta)) / binsize
minDistance = 0.005
maxDistance = 0.01

def distanceFieldFromCloud(cloud):
    shape = np.array((extent - origin) // binsize, dtype = np.int)
    
    distanceField = np.full(shape, 1)
    for c in cloud:
        index = np.array((c - origin) // binsize, dtype=np.int)
        if np.all(index >= 0) and np.all(index < shape):
            distanceField[tuple(index)] = 0
    distanceField = distance_transform_edt(distanceField, binsize)
    return distanceField

def cloudFromMask(mask):
    return ((np.array(np.wh
    ere(mask)) + 0.5) * binsize).T

def kindOflaplacian(distanceField):
    fxx = correlate(distanceField, np.array([
        [[1, -2, 1]]
    ]), mode = 'nearest') / (binsize ** 2)
    fyy = correlate(distanceField, np.array([
        [[1],
        [-2],
        [1]]
    ]), mode = 'nearest') / (binsize ** 2)
    fzz = correlate(distanceField, np.array([
        [[1]], [[-2]], [[1]]
    ]), mode = 'nearest') / (binsize ** 2)
    return fxx, fyy, fzz

def findSkeleton(distanceField):
    fxx, fyy, fzz = kindOflaplacian(distanceField)
    return ((abs(fxx) > threshold) | + (abs(fyy) > threshold) + (abs(fzz) > threshold)) & (distanceField > minDistance) & (distanceField < maxDistance)

def skeletonizer(cloud):
    field = distanceFieldFromCloud(cloud)
    skeleton = findSkeleton(field)
    for i in range(0):
        cloud = cloudFromMask(skeleton)
        field = distanceFieldFromCloud(cloud)
        skeleton = findSkeleton(field)
    return skeleton
