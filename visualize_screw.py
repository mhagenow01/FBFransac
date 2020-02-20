from multiprocessing import Queue, Process
import matplotlib.pyplot as plt
import Verbosifier
from Mesh import Mesh
from ModelFinder import ModelFinder
import numpy as np 
from matplotlib.animation import FuncAnimation
import functools
import json
import point_cloud_utils as pcu
from mpl_toolkits.mplot3d import Axes3D

def findHypotheses(Q : Queue, cloud, cloudNormals):
    mesh = Mesh('Models/ToyScrew-Yellow.stl')
    mesh.compileFeatures(N = 10)
    finder = ModelFinder(mesh)
    for o, r in finder.findInCloud(cloud, cloudNormals):
        faces = (mesh.Faces @ r.T) + o
        Q.put(np.copy(faces))
    return


def showHypotheses(ax, Q:Queue,t):
    while not Q.empty():
        faces = Q.get_nowait()
        ax.scatter(faces[:,0], faces[:,1], faces[:,2], color = 'red')
    return

if __name__ == '__main__':
    
    Verbosifier.enableVerbosity()
    with open('Models/SCrewScene.json') as fin:
        cloud = []
        screwCloud = np.array(json.load(fin))
        for p in screwCloud:
            if not np.any(np.isnan(np.array(p))) and np.linalg.norm(np.array(p)-np.array((0,0,0.4)))< 0.3:
                cloud.append(p)
        cloud = np.array(cloud)
        fullCloud = cloud[np.random.choice(range(len(cloud)), len(cloud))]


    mask = ModelFinder.voxelFilter(fullCloud, size = 0.005)
    cloudNormals = pcu.estimate_normals(fullCloud, 5)
    cloud, cloudNormals = fullCloud[mask], cloudNormals[mask]

    fig = plt.figure()
    ax = fig.gca(projection = '3d')

    Q = Queue()
    ani = FuncAnimation(fig, functools.partial(showHypotheses, ax, Q), range(1), repeat_delay=1)
    ax.scatter(cloud[:,0], cloud[:,1], cloud[:,2])

    process = Process(target = findHypotheses, args = (Q, cloud, cloudNormals))
    process.start()
    plt.show()
    process.terminate()

