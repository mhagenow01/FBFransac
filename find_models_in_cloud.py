import Verbosifier
from Mesh import Mesh
import numpy as np
import json
import matplotlib.pyplot as plt
from ModelFinder import ModelFinder
import open3d as o3d
from mpl_toolkits.mplot3d import Axes3D

def main():
    ''' Given a cloud and meshes to find this will invoke
    the model finder in order to find mesh instances in the scene
    The final result is plotted using Open3d 
    '''

    mesh_files = ['Models/hammer.stl']
    scene = 'Models/Cloud_3_hammers.json'
    gridResolution = 0.001

    Verbosifier.enableVerbosity()

    with open(scene) as fin:
        noisyCloud = np.array(json.load(fin))
        noisyCloud = noisyCloud[np.linalg.norm(noisyCloud, axis=1) < 0.5]

    finder = ModelFinder()
    finder.set_resolution(gridResolution)

    meshes = []
    for mesh_file_temp in mesh_files:
        mesh_temp = Mesh(mesh_file_temp, gridResolution)
        meshes.append(mesh_temp)
    finder.set_meshes(meshes)
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
    for m, pose in instances:
        # TODO: add a way to get the model file from the found instances
        mesh = o3d.io.read_triangle_mesh("Models/ToyScrew-Yellow.stl")
        mesh.rotate(pose[0],center=False) #TODO: check this once all (R,o) stuff is figured out
        mesh.translate(pose[1].reshape((3,))) #TODO: check this once all (R,o) stuff is figured out
        plotting_objects.append(mesh)

    o3d.visualization.draw_geometries(plotting_objects)


if __name__ == '__main__':
    main()