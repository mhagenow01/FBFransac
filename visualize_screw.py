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

def flipNormals(cloudNormals):
    for i,n in enumerate(cloudNormals):
        if n[2] > 0:
            cloudNormals[i] = -n

def main():
    Verbosifier.enableVerbosity()
    with open('Models/Cloud_ToyScrew-Yellow.json') as fin:
        cloud = []
        screwCloud = np.array(json.load(fin))
        for p in screwCloud:
            if not np.any(np.isnan(np.array(p))):
                # if not np.any(np.isnan(np.array(p))) and np.linalg.norm(np.array(p) - np.array((0, 0, 0.4))) < 0.3:
                cloud.append(p)
        cloud = np.array(cloud)
        fullCloud = cloud  # [np.random.choice(range(len(cloud)), len(cloud))]

    cloudNormals = pcu.estimate_normals(fullCloud, 20)
    # mask = ModelFinder.voxelFilter(fullCloud, size = 0.005)
    # cloud, cloudNormals = fullCloud[mask], cloudNormals[mask]
    cloud, cloudNormals = ModelFinder.planarCloudSampling(fullCloud, cloudNormals, radius=0.01)
    flipNormals(cloudNormals)

    fig = plt.figure()
    ax = fig.gca(projection='3d')

    Q = Queue()
    ani = FuncAnimation(fig, functools.partial(showHypotheses, ax, Q), range(1), repeat_delay=1000)
    ax.scatter(cloud[:, 0], cloud[:, 1], cloud[:, 2])
    ax.quiver(cloud[:, 0], cloud[:, 1], cloud[:, 2], cloudNormals[:, 0] * 0.01, cloudNormals[:, 1] * 0.01,
              cloudNormals[:, 2] * 0.01)

    process = Process(target=findHypotheses, args=(Q, cloud, cloudNormals))
    process.start()
    ax.set_xlim3d(-.05, 0.05)
    ax.set_ylim3d(-.05, 0.05)
    ax.set_zlim3d(-0.05, 0.05)
    plt.show()
    process.terminate()

if __name__ == '__main__':
    main()


