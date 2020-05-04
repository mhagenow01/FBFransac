import json
import numpy as np 
import point_cloud_utils as pcu
from PointCloudFromMesh import *
from pykdtree.kdtree import KDTree
from queue import Queue
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D

EPSILON = 0.001
ANGLE_TOLERANCE = 5 * np.pi / 180


class Plane:
    def __init__(self, x, n):
        self.X = x
        self.N = n
    
    def distance(self, r):
        return np.abs(self.signed_distance(r))
    
    def signed_distance(self, r):
        return np.dot(r - self.X, self.N)
    
    def distance_in_plane(self, r):
        d = r - self.X
        return np.linalg.norm(d - self.signed_distance(r) * self.N)
    
    def intersects(self, a, b):
        return np.sign(self.signed_distance(a)) != np.sign(self.signed_distance(b))

    def __repr__(self):
        return f'({self.X};{self.N})'


def compute_convex_set(point, cloud, kd, normals):
    distances, index = kd.query(point, 20)
    index = index[0]
    start = index[0]
    startPoint = cloud[start]
    
    q = Queue()
    q.put((start, start))
    seen = {start}
    planes = []
    indices = []
    while not q.empty():
        last, next = q.get()
        point = cloud[next]
        invalid = False
        for p in planes:
            if p.distance(point) > EPSILON and p.distance(startPoint) > EPSILON \
                and p.intersects(startPoint, point) and p.distance_in_plane(point) < EPSILON:
                print(p.distance(point))
                print(p.distance(startPoint))
                print(p)
                print(point)
                invalid = True
                break

        if invalid:
            continue
        _, index = kd.query(cloud[next:next+1,:], 5)
        indices.append(next)
        newPlane = True
        n = normals[next]
        for p in planes:
            if np.abs(n.dot(p.N)) > 0.9 and np.linalg.norm(point - p.X) < EPSILON:
                newPlane = False
                break
        if newPlane:
            planes.append(Plane(point, normals[next]))
        print(len(planes))
        print(len(indices))
        for i in index[0]:
            if i not in seen:
                seen.add(i)
                q.put((next, i))
    return np.array(indices)

if __name__ == '__main__':
    scene = pointCloudFromMesh('Models/ToyScrew-Yellow.stl', 10000)

    normals = pcu.estimate_normals(scene, 10)
    indices = compute_convex_set(np.array([[1,1,-1]]), scene, KDTree(scene), normals)
    points = scene[indices]
    ax = plt.gca(projection = '3d')
    ax.scatter(points[:,0], points[:,1], points[:,2])
    ax.quiver(points[:,0], points[:,1], points[:,2], normals[indices][:,0], normals[indices][:,1], normals[indices][:,2], length = 0.001)
    plt.show()