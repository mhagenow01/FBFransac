from ModelProfile import ObjectProfile, SupportSphere
import pickle
import sys
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import open3d as o3d


def drawSphere(vis, pos, r):
    mesh = o3d.geometry.TriangleMesh.create_sphere(r)
    mesh.translate(pos.reshape((3,)))
    vis.add_geometry(mesh)

if __name__ == '__main__':
    file = sys.argv[1]
    with open(file, 'rb') as fin:
        profile = pickle.load(fin)
    
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    for r, spheres in profile.KeyPoints:
        for s in spheres:
            drawSphere(vis, s.X, r)
    
    vis.run()

    