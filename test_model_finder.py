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

if __name__ == '__main__':
    Verbosifier.enableVerbosity()
    gridResolution = 0.01
    with open('Models/ComparisonScenes/Cloud_comparison_scene_1.json') as fin:
        scene = np.array(json.load(fin))
    #     #scene = scene[np.linalg.norm(scene, axis = 1) < 0.5]
    # p = Profile()
    # p.enable()
    print(scene.shape)
    finder = ModelFinder()
    finder.set_meshes(['Models/ComparisonSTLs/hammer.stl'], gridResolution)
    finder.set_scene(scene)
    instances = finder.findInstances()
    # p.disable()
    # stats = Stats(p).sort_stats('tottime')
    # stats.print_stats()

    ax = plt.gca(projection = '3d')
    for mesh, (rotation, origin) in instances:
        facePoints = mesh.Faces @ rotation.T + origin
        # print(origin)
        ax.scatter(facePoints[:,0], facePoints[:,1], facePoints[:,2], color = 'red')
    ax.scatter(scene[:,0], scene[:,1], scene[:,2], color = 'blue')
    plt.show()