from Mesh import Mesh
import numpy as np
import itertools
import functools
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D

def angleAndDistance(n1, n2, vertices, verts):
    v1, v2 = vertices[verts[0]], vertices[verts[1]]
    r = v2 - v1
    d = np.linalg.norm(r)
    r /= d
    return np.arccos(abs(n1.dot(r))), np.arccos(abs(n2.dot(r))), d


if __name__ == '__main__':
    mesh = Mesh('Models/ToyScrew-Yellow.stl')
    nFaces = len(mesh.Faces)


    points = []
    sizes = []
    for i in range(nFaces):
        for j in range(i + 1, nFaces):
            f1, f2 = mesh.Faces[[i,j]]
            n1, n2 = mesh.Normals[[i,j]]
            r = f2 - f1

            v_ind1, v_ind2 = mesh.trimesh.faces[[i,j]]
            distances = np.array(list( map(functools.partial(angleAndDistance, n1, n2, mesh.trimesh.vertices), itertools.product(v_ind1, v_ind2)) ))
            minDistace = np.min(distances, axis = 0)
            maxDistance = np.max(distances, axis =0)
            meanDistance = (minDistace + maxDistance) / 2

            points.append(meanDistance)
            sizes.append(maxDistance - minDistace)
    points = np.array(points)
    sizes = np.array(sizes)
    fig = plt.figure()
    ax = fig.gca(projection = '3d')

    #ax.set_aspect(1)
    ax.scatter(points[:,0], points[:,1], points[:,2])
    fig.canvas.draw()
    s = ((ax.get_window_extent().height  / (sizes[:,2]) * 72./fig.dpi) ** 2) / 10000000
    print(s)
    ax.clear()
    ax.scatter(points[:,0], points[:,1], points[:,2], s = np.sqrt(sizes[:,2]))
    plt.show()
    