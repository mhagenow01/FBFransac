from Mesh import Mesh
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
import json
import Verbosifier
from ModelFinder import ModelFinder
from cProfile import Profile
from pstats import Stats
from ModelProfile import *
import itertools
import open3d as o3d
import glob
import sys
from scipy.stats import special_ortho_group
from PointCloudFromMesh import pointCloudFromMesh
import os


def saveResults(fileName, pcd, instances):
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(pcd)

    for m, pose, file in instances:
        mesh = o3d.io.read_triangle_mesh(file)
        mesh.rotate(pose[0],center=False)
        mesh.translate(pose[1].reshape((3,)))
        vis.add_geometry(mesh)

    image = vis.capture_screen_float_buffer(True)
    plt.imsave(fileName, np.asarray(image), dpi = 1)
    vis.destroy_window()


if __name__ == '__main__':
    
    finder = ModelFinder()
    meshFiles = glob.glob(sys.argv[1])
    finder.set_meshes(meshFiles)
    for file in glob.glob(sys.argv[1]):
        R = special_ortho_group.rvs(3)
        finder.set_scene(pointCloudFromMesh(file, 4000) @ R.T)
        instances = finder.findInstances()
        print(instances)
        print(R)
        print(f'Correct Model: {os.path.split(file)[-1]}')
