from Mesh import Mesh
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
import json
import Verbosifier
from ModelFinder import ModelFinder

if __name__ == '__main__':
    Verbosifier.enableVerbosity()
    mesh = Mesh('Models/ToyScrew-Yellow.stl')

    with open('Models\ScrewScene.json') as fin:
        scene = np.array(json.load(fin))
        scene = scene[np.linalg.norm(scene, axis = 1) < 0.5]

    finder = ModelFinder()
    finder.set_resolution(0.001)
    finder.set_meshes([mesh])
    finder.set_scene(scene)
    instances = finder.findInstances()

    ax = plt.gca(projection = '3d')
    for mesh, (origin, rotation) in instances:
        facePoints = (mesh.Faces + origin) @ rotation
        ax.scatter(facePoints[:,0], facePoints[:,1], facePoints[:,2], color = 'red')
    ax.scatter(scene[:,0], scene[:,1], scene[:,2], color = 'blue')
    plt.show()