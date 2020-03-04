from Mesh import Mesh
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time


if __name__ == '__main__':
    mesh = Mesh('Models/ToyScrew-Yellow.stl')
    mesh.cacheMeshDistance(0.001)

    
    centers = np.zeros((mesh.DistanceCache.size, 3))
    for i in range(mesh.DistanceCache.size):
        ijk = np.unravel_index(i, mesh.DistanceCache.shape)
        centers[i] = (np.array(ijk) + 0.5) * mesh.BinSize + mesh.BoundingBoxOrigin

    s = time.time()
    distances = mesh.distanceQuery(centers)
    print(f'Time to cached query: {time.time() - s}')
    
    onMesh = []
    for i in range(len(centers)):
        if distances[i] > -0.001:
            onMesh.append(centers[i])
    onMesh = np.array(onMesh)

    fig = plt.figure()
    ax = plt.gca(projection = '3d')

    ax.scatter(onMesh[:,0], onMesh[:,1], onMesh[:,2])
    plt.show()
    
