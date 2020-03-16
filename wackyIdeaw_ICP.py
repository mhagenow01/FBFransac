from Skeletonizer import skeletonizer, cloudFromMask
import Verbosifier
from sklearn.cluster import DBSCAN
from multiprocessing import Queue, Process
import matplotlib.pyplot as plt
from Mesh import Mesh
from ModelFinder import ModelFinder
import numpy as np
from matplotlib.animation import FuncAnimation
import functools
import json
import point_cloud_utils as pcu
from trimesh.proximity import ProximityQuery
from mpl_toolkits.mplot3d import Axes3D
import time
from ICP_refinement import ICPrefinement

def cluster(skeleton):
    cloud = cloudFromMask(skeleton)
    clustering = DBSCAN(eps=0.01, min_samples=1)
    clustering.fit(cloud)
    cand_points = []

    for ii in range(0,max(clustering.labels_)+1):
        cand_points.append(np.mean(cloud[clustering.labels_==ii],axis=0))

    return cand_points

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
        print("LEN",len(faces_temp))
        ax.set_xlim3d(-.3, 0)
        ax.set_ylim3d(-.2, 0.)
        ax.set_zlim3d(0.2, 0.5)

        for ii in range(0,len(faces_temp)):
            ax.scatter(faces_temp[ii][:, 0], faces_temp[ii][:, 1], faces_temp[ii][:, 2], color='red')

        ax.scatter(cloud[:, 0], cloud[:, 1], cloud[:, 2], color='blue')
    return

def runICP(Q: Queue, cloud, cand_points):
    mesh_one = Mesh('Models/ToyScrew-Yellow.stl')
    time.sleep(3)

    meshes = []
    poses = []

    for ii in range(0,len(cand_points)):
        r_temp = np.eye(3)
        o_temp = cand_points[ii]
        meshes.append(mesh_one.Faces)
        poses.append((r_temp,o_temp))
    icp = ICPrefinement(cloud,meshes,poses)

    count = 0

    while 1:
        faces_temp = []

        for ii in range(0,len(cand_points)):
            faces_temp.append((mesh_one.Faces @ poses[ii][0].T) + poses[ii][1])

        Q.put(np.copy(faces_temp))
        print(poses)

        # Run an iteration of the ICP
        start_time = time.time()
        icp.runICPiteration()
        elapsed_time = time.time() - start_time

        meshes, poses = icp.getUpdatedMeshes()
        time.sleep(2)
        print("ICP Iteration: "+str(count))

        count = count + 1
    return

def run_icp_test(cloud,cand_points,mesh):
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    Q = Queue()
    ani = FuncAnimation(fig, functools.partial(showICP, ax, cloud, Q, mesh), range(1), repeat_delay=1000)

    process = Process(target=runICP, args=(Q, cloud, cand_points))
    process.start()
    ax.set_xlim3d(-.3, 0)
    ax.set_ylim3d(-.2, 0.)
    ax.set_zlim3d(0.2, 0.5)
    plt.show()
    process.terminate()


def main():
    Verbosifier.enableVerbosity()
    mesh = Mesh('Models/ToyScrew-Yellow.stl')

    with open('Models/Cloud_ToyScrew-Yellow.json') as fin:
        realCloud = np.array(json.load(fin))
    with open('Models/ScrewScene.json') as fin:
        noisyCloud = np.array(json.load(fin))
        noisyCloud = noisyCloud[np.linalg.norm(noisyCloud, axis=1) < 0.5]

    print('making scene')
    mask = skeletonizer(noisyCloud)
    cand_points = cluster(mask)

    run_icp_test(realCloud,cand_points,mesh)


if __name__ == '__main__':
    main()