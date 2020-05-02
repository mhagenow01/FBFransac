import Verbosifier
from Mesh import Mesh
import numpy as np
import json
import matplotlib.pyplot as plt
from ModelFinder import ModelFinder
import open3d as o3d
from mpl_toolkits.mplot3d import Axes3D
from ModelProfile import *


def main():
    ''' Given a cloud and meshes to find this will invoke
    the model finder in order to find mesh instances in the scene
    The final result is plotted using Open3d 
    '''

    mesh_files = ['Models/ComparisonSTLs/hammer.stl', 'Models/ComparisonSTLs/saw.stl']
    scene = 'Models/ComparisonScenes/Cloud_comparison_scene_1.json'
    gridResolution = 0.002

    Verbosifier.enableVerbosity()

    with open(scene) as fin:
        noisyCloud = np.array(json.load(fin))

    finder = ModelFinder()
    finder.set_meshes(mesh_files, gridResolution)
    finder.set_scene(noisyCloud)
    instances = finder.findInstances()

    # Plot the Cloud and the meshes using Open3D
    plotting_objects = []

    # Original point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(noisyCloud)
    pcd.paint_uniform_color(np.array([0.2, 0.2, 0.8]).reshape((3,1)))
    plotting_objects.append(pcd)

    # Add in all the found meshes
    for m, pose, file in instances:
        mesh = o3d.io.read_triangle_mesh(file)
        mesh.rotate(pose[0],center=False)
        mesh.translate(pose[1].reshape((3,)))
        plotting_objects.append(mesh)

    o3d.visualization.draw_geometries(plotting_objects)


if __name__ == '__main__':
    main()