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
from trimesh.proximity import ProximityQuery

def findHypotheses(Q : Queue, cloud, cloudNormals):
    Verbosifier.enableVerbosity()
    mesh = Mesh('Models/ToyScrew-Yellow.stl')
    mesh.compileFeatures(N = 10000)
    finder = ModelFinder(mesh)
    bestScore = None
    for o, r in finder.findInCloud(cloud, cloudNormals):
        score = scoreHypothesis(mesh, cloud, o, r)
        if bestScore is None or score > bestScore:
            bestScore = score
            faces = (mesh.Faces @ r.T) + o
            Q.put(np.copy(faces))
    return

def scoreHypothesis(mesh, cloud, o, r):
    distances = ProximityQuery(mesh.trimesh).signed_distance((cloud - o) @ r)
    fitPoints = 0
    for d in distances:
        if abs(d) < 0.002:
            fitPoints += 1
        elif d > 0:
            fitPoints -= 5
    return fitPoints


def showHypotheses(ax, cloud, Q:Queue, t):
    faces = None
    while not Q.empty():
        faces = Q.get_nowait()
    if faces is not None:
        ax.clear()
        
        ax.set_xlim3d(-.05, 0.05)
        ax.set_ylim3d(-.05, 0.05)
        ax.set_zlim3d(-0.05, 0.05)
        ax.scatter(faces[:,0], faces[:,1], faces[:,2], color = 'red')
        ax.scatter(cloud[:,0], cloud[:,1], cloud[:,2], color = 'blue')
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

    cloudNormals = pcu.estimate_normals(fullCloud, 10)
    mask = ModelFinder.voxelFilter(fullCloud, size = 0.005)
    cloud, cloudNormals = fullCloud[mask], cloudNormals[mask]
    #cloud, cloudNormals = ModelFinder.meanPlanarCloudSampling(fullCloud, cloudNormals, 0.01, 0.2, 0.005)
    flipNormals(cloudNormals)

    fig = plt.figure()
    ax = fig.gca(projection='3d')

    Q = Queue()
    meshPoints = ax.scatter([], [], [], color = 'red')
    ani = FuncAnimation(fig, functools.partial(showHypotheses, ax, cloud, Q), range(1), repeat_delay=1000)
    ax.scatter(cloud[:, 0], cloud[:, 1], cloud[:, 2], color = 'blue')
    ax.quiver(cloud[:, 0], cloud[:, 1], cloud[:, 2], cloudNormals[:, 0] * 0.01, cloudNormals[:, 1] * 0.01,cloudNormals[:, 2] * 0.01, color = 'blue')

    process = Process(target=findHypotheses, args=(Q, cloud, cloudNormals))
    process.start()
    ax.set_xlim3d(-.05, 0.05)
    ax.set_ylim3d(-.05, 0.05)
    ax.set_zlim3d(-0.05, 0.05)
    plt.show()
    process.terminate()

if __name__ == '__main__':
    main()


