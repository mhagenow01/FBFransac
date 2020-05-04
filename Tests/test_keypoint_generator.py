import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import json
from KeyPointGenerator import KeyPointGenerator
from Graph import Graph
from Mesh import Mesh


if __name__ == '__main__':
    with open('Models/Cloud_ToyScrew-Yellow-0.0005.json') as fin:
        scene = np.array(json.load(fin))
    kg = KeyPointGenerator(0.000, 0.1, 1, 1)
    KeyPointGenerator.BinSize = 0.001
    mesh = Mesh('Models/ToyScrew-Yellow.stl', KeyPointGenerator.BinSize)
    mesh.cacheMeshDistance()
    #origin, field = kg.distanceFieldFromCloud(scene, kg.BinSize)
    origin, field = mesh.BoundingBoxOrigin, mesh.DistanceCache
    skeleton = kg.findSkeleton(field)

    g = Graph()
    g.fromMatrix(skeleton)
    g.prune(1, 1, 1)

    ax = plt.gca(projection = '3d')
    cloud, _ = g.toCloud(g.Nodes.values())
    cloud = (cloud + 0.5) * KeyPointGenerator.BinSize + origin
    ax.scatter(cloud[:,0], cloud[:,1], cloud[:,2])
    components = g.getComponents()
    print(list(map(len, components)))
    for c in components:
        c.prune(8, 1)
        cloud, distances = c.toCloud(c.Nodes.values())
        print(distances)
        cloud = (cloud + 0.5) * KeyPointGenerator.BinSize + origin
        ax.scatter(cloud[:,0], cloud[:,1], cloud[:,2], s = 80, c = 'red')

    # centers = g.connectedComponentCenters()
    # centerPoints, _ = g.toCloud(centers)
    # centerPoints = (centerPoints + 0.5) * KeyPointGenerator.BinSize + origin


    #ax.scatter(centerPoints[:,0], centerPoints[:,1], centerPoints[:,2], s = 80)
    
    ax.set_xlim([-0.015, 0.015])
    ax.set_ylim([-0.015, 0.015])
    ax.set_zlim([-0.015, 0.03])
    plt.show()