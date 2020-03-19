import Verbosifier
from sklearn.cluster import DBSCAN
from multiprocessing import Queue, Process
import matplotlib.pyplot as plt
from Mesh import Mesh
import numpy as np
from matplotlib.animation import FuncAnimation
import functools
import json
import point_cloud_utils as pcu
from trimesh.proximity import ProximityQuery
from mpl_toolkits.mplot3d import Axes3D
import time
from ICP_refinement import ICPrefinement
from ModelFinder import ModelFinder

gridResolution = 0.001

def flipNormals(cloudNormals):
    for i, n in enumerate(cloudNormals):
        if n[2] > 0:
            cloudNormals[i] = -n

def showICP(ax, cloud, Q: Queue, t, mesh):
    faces_temp = None
    while not Q.empty():
        faces_temp = Q.get_nowait()

    if faces_temp is not None:
        ax.clear()
        print("LEN",faces_temp.shape)
        ax.set_xlim3d(-0.1, 0.1)
        ax.set_ylim3d(-0.1, 0.1)
        ax.set_zlim3d(-0.1, 0.1)

        for ii in range(0,len(faces_temp)):
            ax.scatter(faces_temp[ii][:, 0], faces_temp[ii][:, 1], faces_temp[ii][:, 2], color='red')

        ax.scatter(cloud[:, 0], cloud[:, 1], cloud[:, 2], color='blue')
    return

def runICP(Q: Queue, cloud):
    Verbosifier.enableVerbosity()
    gridResolution = 0.001
    mesh_one = Mesh('Models/ToyScrew-Yellow.stl',gridResolution)
    
    finder = ModelFinder()
    finder.set_resolution(gridResolution)
    finder.set_meshes([mesh_one])
    finder.set_scene(cloud)
    instances = finder.findInstances()
    time.sleep(3)

    meshes = []
    poses = []

    for m, sceneKp, meshKp in instances:
        meshes.append(m.Faces - meshKp)
        poses.append((np.eye(3), sceneKp))
    icp = ICPrefinement(cloud,meshes,poses)

    count = 0

    while 1:
        faces_temp = []

        for m, pose in zip(meshes, poses):
            faces_temp.append((m @ pose[0].T) + pose[1])

        Q.put(np.copy(faces_temp))
        print(poses)

        # Run an iteration of the ICP
        start_time = time.time()
        icp.runICPiteration()
        elapsed_time = time.time() - start_time

        meshes, poses = icp.getUpdatedMeshes()
        # time.sleep(2)
        print("ICP Iteration: "+str(count))

        count = count + 1
    return

def run_icp_test(cloud,mesh):
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    Q = Queue()
    ani = FuncAnimation(fig, functools.partial(showICP, ax, cloud, Q, mesh), range(1), repeat_delay=50)

    process = Process(target=runICP, args=(Q, cloud))
    process.start()
    ax.set_xlim3d(-.3, 0)
    ax.set_ylim3d(-.2, 0.)
    ax.set_zlim3d(0.2, 0.5)
    plt.show()
    process.terminate()


def main():
    Verbosifier.enableVerbosity()
    mesh = Mesh('Models/ToyScrew-Yellow.stl', gridResolution)

    with open('Models/Cloud_hand_and_screw_simulated.json') as fin:
        noisyCloud = np.array(json.load(fin))
        noisyCloud = noisyCloud[np.linalg.norm(noisyCloud, axis=1) < 0.5]

    run_icp_test(noisyCloud,mesh)


if __name__ == '__main__':
    main()