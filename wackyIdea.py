from Mesh import Mesh
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
import json
from scipy.signal import convolve2d
from scipy.ndimage import gaussian_filter
from scipy.ndimage.filters import correlate
from scipy import ndimage
#from matplotlib.animation import ArtistAnimation


origin = np.array((-0.03, -0.03, -0.03))
extent = np.array((0.03, 0.03, 0.03))
binsize = 0.0005
theta = np.pi / 3
def distanceFieldFromCloud(cloud):
    # origin = np.min(cloud, axis = 0) - 30 * binsize
    # extent = np.max(cloud, axis = 0) + 30 * binsize
    shape = np.array((extent - origin) // binsize, dtype = np.int)
    
    distanceField = np.full(shape, 1)
    for c in cloud:
        index = np.array((c - origin) // binsize, dtype=np.int)
        if np.all(index >= 0) and np.all(index < shape):
            distanceField[tuple(index)] = 0
    distanceField = ndimage.distance_transform_edt(distanceField, binsize)
    return distanceField

def derivatives(cloud):
    Z = distanceFieldFromCloud(cloud)
    print('distance field loaded')
    zx = correlate(Z, np.array([
        [[-100, -0, 100]]
    ]))[1:-1,1:-1]
    zy = correlate(Z, np.array([
        [[-1],
        [-0],
        [1]]
    ]))[1:-1,1:-1] / (2 * binsize)
    zz = correlate(Z, np.array([
        [[-1]], [[-0]], [[1]]
    ]))[1:-1,1:-1] / (2 * binsize)

    zxx = correlate(zx, np.array([
        [[-1, -0, 1]]
    ])) / (2 * binsize)
    zyy = correlate(zy, np.array([
        [[-1],
        [-0],
        [1]]
    ])) / (2 * binsize)
    zzz = correlate(zz, np.array([
        [[-1]], [[-0]], [[1]]
    ])) / (2 * binsize)

    return Z, zx, zy, zz, zxx, zyy, zzz


def loadCloud2d(name):
    with open(name) as fin:
        cloud = np.array(json.load(fin))
        cloud = cloud[(cloud[:,0] < 0.001) & (cloud[:,0] > -0.001)]
        cloud = cloud[cloud[:,1] > 0]
        cloud = cloud[:,1:]
    return cloud


def cloudFromDistanceField(field):
    cloud = []
    for i in range(field.size):
        ijk = np.unravel_index(i, field.shape)
        if field[ijk] == 1:
            position = (np.array(ijk) + 0.5) * binsize
            cloud.append(position)
    return np.array(cloud)


if __name__ == '__main__':
    mesh = Mesh('Models/ToyScrew-Yellow.stl')

    # mesh.cacheMeshDistance(0.001)
    # Z = mesh.DistanceCache[mesh.DistanceCache.shape[0]//2, :,:]
    # Z = np.abs(Z)

    # realCloud = loadCloud2d('Models\Cloud_ToyScrew-Yellow.json')
    # noisyCloud = loadCloud2d('Models\Cloud_ToyScrew-Yellow-0.0005.json')
    with open('Models\Cloud_ToyScrew-Yellow.json') as fin:
        realCloud = np.array(json.load(fin))
    with open('Models\Cloud_ToyScrew-Yellow-0.0005.json') as fin:
        noisyCloud = np.array(json.load(fin))

    z, zx, zy, zz, zxx, zyy, zzz = derivatives(realCloud)
    nz, nzx, nzy, nzz, nzxx, nzyy, nzzz = derivatives(realCloud)

    level = np.zeros_like(zxx)
    # This should be the approximate divergence across 
    # a medial axis between two faces with an angle of (pi - theta)
    # between them.
    threshold = np.sqrt(2 - 2 * np.cos(theta)) / binsize
    mask = (abs(nzxx + nzyy + nzzz) > threshold) & (abs(nz[1:-1,1:-1]) > 0.0015) #& (abs(nz[1:-1,1:-1]) < 0.0015)
    level[mask] = 1
    #mask = (abs(zxx + zyy) > threshold) & (abs(z[1:-1,1:-1]) > 0.002) & (abs(z[1:-1,1:-1]) < 0.01)
    #level[mask] = -1
    print(np.sum(level))
    cloud = cloudFromDistanceField(level)
    ax = plt.gca(projection = '3d')
    ax.scatter(cloud[:,0], cloud[:,1], cloud[:,2])
    plt.figure()
    plt.hist(abs(nzxx + nzyy + nzzz).flatten(), bins = 200)
    #ax.quiver(zx * 100, zy * 100, headwidth = 3, headlength = 5, minlength = 0.001, units = 'xy')
    plt.show()