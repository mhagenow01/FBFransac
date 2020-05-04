import Verbosifier
from Mesh import Mesh
import numpy as np
import json
from ModelFinder import ModelFinder
import matplotlib.pyplot as plt
import open3d as o3d
from mpl_toolkits.mplot3d import Axes3D
from ModelProfile import *
import time


def main():
    ''' Given a cloud and meshes to find this will invoke
    the model finder in order to find mesh instances in the scene
    The final result is plotted using Open3d 
    '''

    mesh_files = ['Models/ComparisonSTLs/wrench.stl']
    scene = 'Models/ComparisonScenes/Cloud_comparison_scene_1.json'
    gridResolution = 0.01
    # gridResolution = 0.1

    Verbosifier.enableVerbosity()

    with open(scene) as fin:
        noisyCloud = np.array(json.load(fin))

    finder = ModelFinder()


    finder.set_meshes(mesh_files,gridResolution)
    mesh_temp = Mesh(mesh_files[0],gridResolution)
    # print("HELLO")
    finder.set_scene(noisyCloud)

    r = np.array([[1.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0],
                       [0.0, 0.0, 1.0]])

    # r = np.array([[-0.63071587,  0.73215234,  0.25719727],
    #               [0.27241566,  0.51923612, -0.81005158],
    #               [-0.72662727, -0.44084783, -0.52694022]])

    # o = np.array([-0.08, 0.2, 0.005]) # Hammer
    # o = np.array([-0.2, -0.08, 0.08]) # Saw
    # o = np.array([0.13, -0.25, 0.15])  # Pliers
    # o = np.array([0.0, -0.22, 0.0])  # Screw
    # o = np.array([-0.1, -0.08, -0.1])  # Screwdriver - upside down constantly!!
    # o = np.array([0.0, 0.0, 0.0])  # Spanning wrench
    o = np.array([0.15, 0.1, -0.05])  # Wrench


    error = np.inf
    t0 = time.time()
    r,o,error = finder.ICPrandomRestarts(r,o,mesh_temp.Faces,mesh_temp.Normals,mesh_temp.Sizes)
    t1 = time.time()
    print("TIME TAKING",t1-t0)
    # r,o,error = finder.runICP(r,o,mesh_temp.Faces,mesh_temp.Normals,mesh_temp.Sizes)
    print("ERROR:",error)

    # Plot the Cloud and the meshes using Open3D
    plotting_objects = []

    # Original point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(noisyCloud)
    pcd.paint_uniform_color(np.array([0.2, 0.2, 0.8]).reshape((3,1)))
    plotting_objects.append(pcd)


    # # Face points
    # face_points = mesh_temp.Faces @ r.T + o
    # pcd2 = o3d.geometry.PointCloud()
    # pcd2.points = o3d.utility.Vector3dVector(face_points)
    # pcd2.paint_uniform_color(np.array([0.2, 0.8, 0.2]).reshape((3, 1)))
    # # plotting_objects.append(pcd2)

    # # Add in all the found meshes
    # print("O:",o)
    # print("R:",r)

    # TODO: add a way to get the model file from the found instances
    mesh = o3d.io.read_triangle_mesh(mesh_files[0])
    mesh.rotate(r,center=False) #TODO: check this once all (R,o) stuff is figured out
    mesh.translate(o.reshape((3,)))  # TODO: check this once all (R,o) stuff is figured out
    plotting_objects.append(mesh)

    o3d.visualization.draw_geometries(plotting_objects)


if __name__ == '__main__':
    main()