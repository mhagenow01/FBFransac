import numpy as np
from pykdtree.kdtree import KDTree
import itertools
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D
import json
from cProfile import Profile
from pstats import Stats
import line_profiler
import point_cloud_utils as pcu


class Node:
    SceneCloud = None
    SceneNormals = None
    SceneKdTree = None
    MinimumSize = None
    MinimumDistance = None
    MaximumDistance = None
    Directions = np.array([np.array(d) for d in itertools.product(*( ((-1,1),) * 3 ))])
    CardinalDirections = np.array([np.array(d) for d in itertools.product(*( ((-1, 0, 1),) * 3 )) if np.count_nonzero(d) == 1])
    
    def __init__(self, center, size):
        self.Children = None
        self.Center = center
        self.Size = size
        self.IsMedial = False
        self.Value = None
    
    def initChildren(self):
        self.Children = np.empty(8, dtype = Node)
        childSize = self.Size / 2
        childPoints = (self.Center + self.Directions * childSize / 2)
        for i in range(8):
            self.Children[i] = Node(childPoints[i].reshape((1,3)), childSize)

    def tryExpand(self):
        distances, closest = self.SceneKdTree.query(self.Center, k = 1)
        p1 = self.SceneCloud[closest[0]]
        
        if np.any(np.abs(p1 - self.Center) > (self.MaximumDistance + self.Size / 2)):
            return
        if np.all((np.abs(p1 - self.Center) + self.Size / 2) < self.MinimumDistance):
            return

        # If we are at a leaf
        if np.any(self.Size / 2 < self.MinimumSize):
            # Don't flag leaves that actually contain surface 
            # points as being part of the medial axis
            if distances[0] < self.MinimumDistance:
                return
            
            self.IsMedial, self.Value = self.containsMinimum()
        elif self.contains(p1) or self.containsMinimum():
            self.initChildren()
            for c in self.Children:
                c.tryExpand()
   
    def containsMinimum(self):
        childPoints = (self.Center + self.CardinalDirections * self.Size / 2)
        distances, closest = self.SceneKdTree.query(childPoints, k = 1)
        closest = np.array(closest)

        points = self.SceneCloud[closest]
        normals = self.SceneNormals[closest]
        relative = points - childPoints
        backwardsMask = np.sum(relative * normals, axis = 1) < 0
        normals[backwardsMask] = -normals[backwardsMask]
        divergence = np.sum(self.CardinalDirections * normals)

        #print(divergence)
        return divergence > 0.4, divergence

    
    def contains(self, p):
        return np.all(np.abs(p - self.Center) <= self.Size / 2)

    def toCloud(self, cloud, values):
        if self.Children is None:
            if self.IsMedial:
                cloud.append(self.Center.reshape((3,)))
                values.append(self.Value)
        else:
            for c in self.Children:
                c.toCloud(cloud, values)


class MedialAxis:
    def __init__(self, minDistance, maxDistance, resolution, cloud):
        origin = np.min(cloud, 0).reshape((1,3)) - 0.01
        extent = np.max(cloud, 0).reshape((1,3)) + 0.01
        Node.MinimumDistance = minDistance
        Node.MaximumDistance = maxDistance
        Node.MinimumSize = resolution
        Node.SceneCloud = cloud
        Node.SceneNormals = pcu.estimate_normals(cloud, 10, 5)
        Node.SceneKdTree = KDTree(cloud)
        self.Root = Node((origin + extent) / 2, extent - origin)
        self.Root.tryExpand()

    def toCloud(self):
        
        cloud = []
        values = []
        self.Root.toCloud(cloud, values)
        return np.array(cloud), np.array(values)


if __name__ == '__main__':
    with open('Models/Cloud_assorted_objects.json') as fin:
        scene = np.array(json.load(fin))
    
    # p = Profile()
    # p.enable()
    axis = MedialAxis(0.003, 0.006, 0.0005, scene)
    # p.disable()
    # stats = Stats(p).sort_stats('cumtime')
    # stats.print_stats()
    axisCloud, colors = axis.toCloud()

    ax = plt.gca(projection = '3d')
    ax.scatter(axisCloud[:,0], axisCloud[:,1], axisCloud[:,2], c = colors)
    plt.show()
