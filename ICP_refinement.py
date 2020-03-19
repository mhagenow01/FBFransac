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
from pykdtree.kdtree import KDTree
from EfficientRANSAC.scripts.EfficientRANSACFinder import EfficientRANSACFinder as ERF

"""
This Class implements a version of the iterative closest point algorithm
as described in http://evlm.stuba.sk/APLIMAT2018/proceedings/Papers/0876_Prochazkova_Martisek.pdf

It takes a point cloud and candidate models/poses as input
and returns a refined set of models/poses

"""


class ICPrefinement:

    cloud = None
    cloudKdtree = None
    meshes = None
    poses = None
    distance_threshold = 0.02

    tolerance = 1e-4

    def __init__(self,cloud,meshes,poses):
        self.cloud = cloud
        self.meshes = meshes
        self.poses = poses
        self.distance_threshold = 0.02 # 2mm to start
        self.cloudKdtree = KDTree(cloud)

    def runICPiteration(self):


        for kk in range(0,len(self.meshes)):
            mesh = self.meshes[kk]
            pose = self.poses[kk]

            # Compute the nearest point in the point cloud for each point in the model

            face_points = mesh @ pose[0] + pose[1].reshape((1,3))
            distances, closest_indices = self.cloudKdtree.query(face_points, 1)
            closest_points = self.cloud[closest_indices]

            closeEnough = distances < self.distance_threshold
            s_vals = closest_points[closeEnough]
            m_vals = face_points[closeEnough]

            # Add weights to the pairs of points (SKIP FOR NOW)
            # Can be tuned based on things like normals, etc.

            # Calculate R and T using least squares SVD
            # Calculate centroids
            # print(s_vals.shape, m_vals.shape)
            centroid_s = np.mean(s_vals, 0)
            centroid_m = np.mean(m_vals, 0)
            s_vals -= centroid_s
            m_vals -= centroid_m
            S = s_vals.T @ m_vals
            U, sigma, Vh = np.linalg.svd(S)
            V = np.transpose(Vh)

            # Rotation using SVD
            R = np.matmul(V,np.transpose(U))
            t = (centroid_m.reshape((3,1))- R @ centroid_s.reshape((3,1)))

            # REMOVE
            # t = np.zeros((3,1))
            # print ("R in ICP:", R)
            # print ("T in ICP:", t)

            # Update poses - NOTE: translation is with respect to current translation of mesh
            self.poses[kk] = ((R.T @ pose[0].T,pose[1].reshape((3,))-t.reshape((3,))))

            # TODO: Store the most recent error and only run if above a tolerance

            # Compute the summed error E(R,t)
            # Skip this for now...
            error = 0.001


        return error

    def getUpdatedMeshes(self):
        return self.meshes,self.poses


"""

Testing functions for ICP Algorithm

"""

def runICP(Q: Queue, cloud):
    mesh_one = Mesh('Models/ToyScrew-Yellow.stl', 0.001)

    time.sleep(2)

    r1 = np.array([[0.9505638, -0.0953745, 0.2955202]
                                    , [0.1562609, 0.9693090, -0.1897961]
                                    , [-0.2683487, 0.2265915, 0.9362934]])
    r1 = np.array([[1.0, 0.0, 0.0],[0.0, 0.0 , -1.0],[0.0, 1.0, 0.0]])


    o1 = np.array([-0.0145,-0.16150001,0.37150003])


    # r2 = np.array([[0.8383867, -0.2593434, 0.4794255]
    #                   , [ 0.2483189, 0.9647081, 0.0876121]
    #                   , [-0.4852273, 0.0455976, 0.8731983]])
    # o2 = np.array([0.11, -0.02, 0.04])
    #
    # r3 = np.array([[0.8383867, -0.2593434, 0.4794255]
    #                   , [0.2483189, 0.9647081, 0.0876121]
    #                   , [-0.4852273, 0.0455976, 0.8731983]])
    # o3 = np.array([-0.12, -0.02, 0.04])


    # r1 = np.eye(3)
    # r2 = np.eye(3)
    # r3 = np.eye(3)
    # o1 = np.array([0.0, 0.0, 0.0])
    # o2 = np.array([-0.11, 0.0, 0.0])
    # o3 = np.array([0.1, 0.0, 0.0])



    # Create the list of "rough guesses from the FBF-RANSAC
    meshes = []
    meshes.append(mesh_one.Faces)
    # meshes.append(mesh_one.Faces)
    # meshes.append(mesh_one.Faces)

    poses = []
    poses.append((r1,o1))
    # poses.append((r2,o2))
    # poses.append((r3,o3))

    icp = ICPrefinement(cloud,meshes,poses)

    count = 0

    print("SHAPE",np.shape(mesh_one.Faces))


    while 1:
        # faces = ((mesh_one.Faces @ r1.T) + o1,(mesh_one.Faces @ r2.T) + o2,(mesh_one.Faces @ r3.T) + o3)
        faces = ((mesh_one.Faces @ r1.T) + o1,None,None)
        Q.put(np.copy(faces))
        time.sleep(1)
        # Run an iteration of the ICP
        icp.runICPiteration()
        meshes, poses = icp.getUpdatedMeshes()
        r1 = poses[0][0]
        o1 = poses[0][1]
        # r2 = poses[1][0]
        # o2 = poses[1][1]
        # r3 = poses[2][0]
        # o3 = poses[2][1]
        count = count + 1
        print("ICP Iteration: "+str(count))
        print("Screw 1:"+str(o1))
        print(r1)
        # print("Screw 2:"+str(o2))
        # print("Screw 3:"+str(o3))
        print("")


    return


def showICP(ax, cloud, Q: Queue, t):
    faces1 = None
    faces2 = None
    faces3 = None
    while not Q.empty():
        (faces1,faces2,faces3) = Q.get_nowait()
    if faces1 is not None:
        ax.clear()

        ax.set_xlim3d(-.05, 0)
        ax.set_ylim3d(-.18, -0.13)
        ax.set_zlim3d(0.36, 0.41)
        ax.scatter(faces1[:, 0], faces1[:, 1], faces1[:, 2], color='red')
        # ax.scatter(faces2[:, 0], faces2[:, 1], faces2[:, 2], color='red')
        # ax.scatter(faces3[:, 0], faces3[:, 1], faces3[:, 2], color='red')
        ax.scatter(cloud[:, 0], cloud[:, 1], cloud[:, 2], color='blue')
    return


def flipNormals(cloudNormals):
    for i, n in enumerate(cloudNormals):
        if n[2] > 0:
            cloudNormals[i] = -n


def run_icp_test():

    # Get a scene as a starting point
    with open('Models/ScrewScene.json') as fin:
        cloud = np.array(json.load(fin))
        cloud = cloud[np.linalg.norm(cloud, axis=1) < 0.5]

    # Process the scene
    # cloudNormals = pcu.estimate_normals(fullCloud, k=10,smoothing_iterations=3)
    # mask = ERF.voxelFilter(fullCloud, size = 0.005)
    # cloud, cloudNormals = fullCloud[mask], cloudNormals[mask]
    # #cloud, cloudNormals = ModelFinder.meanPlanarCloudSampling(fullCloud, cloudNormals, 0.01, 0.2, 0.005)
    # flipNormals(cloudNormals)
    #
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    #
    Q = Queue()
    meshPoints = ax.scatter([], [], [], color = 'red')
    ani = FuncAnimation(fig, functools.partial(showICP, ax, cloud, Q), range(1), repeat_delay=1000)
    ax.scatter(cloud[:, 0], cloud[:, 1], cloud[:, 2], color = 'blue')
    # ax.quiver(cloud[:, 0], cloud[:, 1], cloud[:, 2], cloudNormals[:, 0] * 0.01, cloudNormals[:, 1] * 0.01,cloudNormals[:, 2] * 0.01, color = 'blue')
    #


    process = Process(target=runICP, args=(Q, cloud))
    process.start()
    ax.set_xlim3d(-.3, 0)
    ax.set_ylim3d(-.2, 0.)
    ax.set_zlim3d(0.2, 0.5)
    plt.show()
    process.terminate()

def test_ICP_alg():

    # Get a scene as a starting point
    with open('Models/Cloud_ToyScrew-Yellow.json') as fin:
        cloud = []
        screwCloud = np.array(json.load(fin))
        for p in screwCloud:
            if not np.any(np.isnan(np.array(p))):
                # if not np.any(np.isnan(np.array(p))) and np.linalg.norm(np.array(p) - np.array((0, 0, 0.4))) < 0.3:
                cloud.append(p)
        cloud = np.array(cloud)
        fullCloud = cloud  # [np.random.choice(range(len(cloud)), len(cloud))]

    # Process the scene
    cloudNormals = pcu.estimate_normals(fullCloud, k=10,smoothing_iterations=3)
    mask = ERF.voxelFilter(fullCloud, size = 0.005)
    cloud, cloudNormals = fullCloud[mask], cloudNormals[mask]
    flipNormals(cloudNormals)

    # Get the mesh
    mesh = Mesh('Models/ToyScrew-Yellow.stl', 0.001)

    r1 = np.eye(3)
    o1 = np.array([0., 0., 0.])

    # Create the list of "rough guesses from the FBF-RANSAC
    meshes = []
    meshes.append(mesh.Faces)

    poses = []
    poses.append((r1, o1))

    icp = ICPrefinement(cloud, meshes, poses)
    count = 0
    for ii in range(0,100):
        icp.getUpdatedMeshes()
        meshes, poses = icp.getUpdatedMeshes()
        r1 = poses[0][0]
        o1 = poses[0][1]

        print("ICP Iteration: " + str(count))
        print("Screw 1:" + str(o1))
        print(r1)
        count = count + 1
        icp.runICPiteration()



if __name__ == '__main__':
    run_icp_test()

