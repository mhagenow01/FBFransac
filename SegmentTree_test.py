import matplotlib.pyplot as plt 
import numpy as np
from cProfile import Profile
import pstats
import functools
import itertools
import time
from Mesh import Mesh
from SegmentTree import SegmentTree
import Verbosifier



###### Test Methods ######

def timedRandomSet():
    d = 20
    n = 1000
    e = 0.01
    error = np.array((e) * d)
    intervals = [
        [n - error, n + error] for n in np.random.randn(n,d)
    ]
    intervals = np.array(intervals)
    pr = Profile()
    pr.enable()
    s = time.time()
    tree = SegmentTree(intervals)
    pr.disable()
    stats = pstats.Stats(pr).sort_stats('tottime')
    stats.print_stats()

    print(f'Compile time with {n}-{d} dimensional boxes of size {e}: {time.time() - s}')
    s = time.time()
    for q in np.random.randn(10000,d):
        tree.containsPoint(q)
    print(f'Query time 10k points: {time.time() - s}')


def imshowTree(tree):
    plt.figure()
    x = np.arange(-4, 4, 0.01)
    y = np.arange(-4, 4, 0.01)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)
    for i, _x in enumerate(x):
        for j, _y in enumerate(y):
            Z[i,j] = len(tree.stab(np.array((_x, _y))))
    plt.imshow(Z.T, origin = 1, extent=[-4,4,-4,4])

def showSimple():
    dx = 0.1
    intervals = [
        [[x,x], [x+2*dx, x+2*dx]] for x in np.arange(0, 4, dx)
    ]
    intervals = np.array(intervals)
    tree = SegmentTree(intervals)

    fig = plt.figure()
    ax = fig.gca()
    ax.set_xlim([-0,6])
    ax.set_ylim([-0,6])
    tree.renderTo2d(ax)
    #imshowTree(tree)
    plt.show()


def angleAndDistance(n1, n2, vertices, verts):
    v1, v2 = vertices[verts[0]], vertices[verts[1]]
    r = v2 - v1
    d = np.linalg.norm(r)
    r /= d
    return np.arccos(abs(n1.dot(r))), np.arccos(abs(n2.dot(r))), d


def testAngleDistanceFeature():
    mesh = Mesh('Models/ToyScrew-Yellow.stl')
    nFaces = len(mesh.Faces)

    intervals = []
    for i in range(nFaces):
        for j in range(i + 1, nFaces):
            f1, f2 = mesh.Faces[[i,j]]
            n1, n2 = mesh.Normals[[i,j]]
            r = f2 - f1

            v_ind1, v_ind2 = mesh.trimesh.faces[[i,j]]
            distances = np.array(list( map(functools.partial(angleAndDistance, n1, n2, mesh.trimesh.vertices), itertools.product(v_ind1, v_ind2)) ))
            minDistace = np.min(distances, axis = 0)
            maxDistance = np.max(distances, axis = 0)
            if np.any(np.isnan(minDistace)) or np.any(np.isnan(maxDistance)):
                continue
            intervals.append([minDistace, maxDistance])
    intervals = np.array(intervals)
    intervals = intervals[np.random.choice(range(len(intervals)), 400)]
    s = time.time()
    tree = SegmentTree(intervals)
    print(f'Feature compile time: {time.time() - s}')
    s = time.time()
    for q in (np.random.rand(10000,3) * np.array((np.pi/2, np.pi/2, 0.05))):
        tree.containsPoint(q)
    print(f'Query time 10k points: {time.time() - s}')

    
if __name__ == '__main__':
    Verbosifier.enableVerbosity()
    #testAngleDistanceFeature()
    #showSimple()
    timedRandomSet()
