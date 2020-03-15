from Mesh import Mesh
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
import json
from Skeletonizer import skeletonizer, cloudFromMask


def plotMedialPoints(skeleton):
    cloud = cloudFromMask(skeleton)
    print(len(cloud))

    ax = plt.gca(projection = '3d')
    ax.scatter(cloud[:,0], cloud[:,1], cloud[:,2])
    ax.set_xlim([-0.3, 0])
    ax.set_ylim([-0.2, 0])
    ax.set_zlim([0.2, 0.5])
    return mask

if __name__ == '__main__':
    mesh = Mesh('Models/ToyScrew-Yellow.stl')

    with open('Models\Cloud_ToyScrew-Yellow.json') as fin:
        realCloud = np.array(json.load(fin))
    with open('Models\ScrewScene.json') as fin:
        noisyCloud = np.array(json.load(fin))
        noisyCloud = noisyCloud[np.linalg.norm(noisyCloud, axis = 1) < 0.5]

    print('making scene')
    print(noisyCloud.shape)
    mask = skeletonizer(noisyCloud)
    plotMedialPoints(mask)

    ax = plt.gca(projection = '3d')
    ax.scatter(noisyCloud[:,0], noisyCloud[:,1], noisyCloud[:,2])
    plt.show()