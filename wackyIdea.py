from Mesh import Mesh
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
import json
from Skeletonizer import skeletonizer, cloudFromMask


def plotMedialPoints(skeleton):
    cloud = cloudFromMask(skeleton)

    ax = plt.gca(projection = '3d')
    ax.scatter(cloud[:,0], cloud[:,1], cloud[:,2])
    return mask

if __name__ == '__main__':
    mesh = Mesh('Models/ToyScrew-Yellow.stl')

    with open('Models\Cloud_ToyScrew-Yellow.json') as fin:
        realCloud = np.array(json.load(fin))
        #realCloud = realCloud[realCloud[:,0] > 0]
    with open('Models\Cloud_ToyScrew-Yellow-0.0005.json') as fin:
        noisyCloud = np.array(json.load(fin))
        noisyCloud = noisyCloud[noisyCloud[:,0] > 0]

    mask = skeletonizer(realCloud)
    plotMedialPoints(mask)
    mask = skeletonizer(noisyCloud)
    plotMedialPoints(mask)



    #plt.figure()
    # ax = plt.gca(projection = '3d')
    # colors = np.array(('r', 'g', 'b', 'yellow'))
    # for i in range(1, n):
    #     mask = labels == i
    #     if np.sum(mask) > 5:
    #         extended = extendedCloud(mask, distance, div)
    #         ax.scatter(extended[:,0], extended[:,1], extended[:,2], s = 30)

    plt.show()