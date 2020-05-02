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
    resultsFolder = os.path.abspath('./Evaluation')
    if not os.path.isdir(resultsFolder):
        os.mkdir(resultsFolder)
    modelNames = [
        'hammer',
        'pliers',
        'saw',
        'screw',
        'screwdriver',
        'spanningwrench',
        'wrench'
    ]
    for sceneId in [3]:
        file = f'Models/ComparisonScenes/Cloud_comparison_scene_{sceneId}.json'
        with open(file) as fin:
            scene = np.array(json.load(fin))
        
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(scene)
        pcd.paint_uniform_color(np.array([0.2, 0.2, 0.8]).reshape((3,1)))

        for combo in itertools.combinations(modelNames, 3):
            print(sceneId, combo)
            
            finder = ModelFinder()
            meshFiles = [f'Models/ComparisonSTLs/{c}.stl' for c in combo]
            finder.set_meshes(meshFiles)
            finder.set_scene(scene)
            instances = finder.findInstances()

            saveResults(
                os.path.join(resultsFolder, '{}_{}_{}-{}.png'.format(*combo, sceneId)),
                pcd, 
                instances
            )