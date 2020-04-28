import Verbosifier
from Mesh import Mesh
import numpy as np
import json
from ModelFinder import ModelFinder
import matplotlib.pyplot as plt
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
    gridResolution = 0.01

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

    r = np.array([[1.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0],
                       [0.0, 0.0, 1.0]])

    # r = np.array([[-0.63071587,  0.73215234,  0.25719727],
    #               [0.27241566,  0.51923612, -0.81005158],
    #               [-0.72662727, -0.44084783, -0.52694022]])

    # These are for the screw scene!!! (-0.02, -0.150, 0.38)
    # o = np.array([-0.02, -0.150, 0.40])
    o = np.array([0.0, 0.0, 0.0])

    r,o,error = finder.ICPrandomRestarts(r,o,mesh_temp.Faces,mesh_temp.Normals,mesh_temp.Sizes)
    # r,o,error = finder.runICP(r,o,mesh_temp.Faces,mesh_temp.Normals,mesh_temp.Sizes)

    print("ERROR:",error)

    # Plot the Cloud and the meshes using Open3D
    plotting_objects = []

    # Original point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(noisyCloud)
    pcd.paint_uniform_color(np.array([0.2, 0.2, 0.8]).reshape((3,1)))
    plotting_objects.append(pcd)


    # Face points
    face_points = mesh_temp.Faces @ r.T + o
    pcd2 = o3d.geometry.PointCloud()
    pcd2.points = o3d.utility.Vector3dVector(face_points)
    pcd2.paint_uniform_color(np.array([0.2, 0.8, 0.2]).reshape((3, 1)))
    plotting_objects.append(pcd2)

    # Add in all the found meshes
    print("O:",o)
    print("R:",r)

    # TODO: add a way to get the model file from the found instances
    mesh = o3d.io.read_triangle_mesh("Models/hammer.stl")
    mesh.rotate(r,center=False) #TODO: check this once all (R,o) stuff is figured out
    mesh.translate(o.reshape((3,)))  # TODO: check this once all (R,o) stuff is figured out
    plotting_objects.append(mesh)

    o3d.visualization.draw_geometries(plotting_objects)


if __name__ == '__main__':
    main()