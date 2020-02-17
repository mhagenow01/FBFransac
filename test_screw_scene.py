from Octree import Octree
import numpy as np
import matplotlib.pyplot as plt
import itertools
import json
import point_cloud_utils as pcu
from pykdtree.kdtree import KDTree
from trimesh.proximity import ProximityQuery
from scipy.spatial.transform import Rotation
from mpl_toolkits.mplot3d import Axes3D
from Mesh import Mesh

def planarCloudSampling(cloud, cloudNormals, radius = 0.1, normalThreshold = 0.1, coplanarThreshold = 0.01):
    print('Sampling cloud!')
    sampledPoints = []
    sampledNormals = []
    for p, n in zip(cloud, cloudNormals):
        represented = False
        for p2, n2 in zip(sampledPoints, sampledNormals):
            if abs(n.dot(n2)) < 1 - normalThreshold:
                continue
            p_r = p - p2
            if abs(p_r.dot(n2)) > coplanarThreshold:
                continue
            p_n2 = p_r - p_r.dot(n2) * n2
            if np.linalg.norm(p_r) > radius:
                continue
            represented = True
            break
            
        if not represented:
            sampledPoints.append(p)
            sampledNormals.append(n)
    return np.array(sampledPoints), np.array(sampledNormals)

def generateHypotheses(cloud, mesh : Mesh):
    print('Finding Hypotheses!')
    cloudNormals = pcu.estimate_normals(fullCloud, 5)
    cloud, cloudNormals = planarCloudSampling(fullCloud, cloudNormals, 0.1, 0.3, 0.001)
    sceneTree = KDTree(cloud)
    r = mesh.Radius

    indexes = list(range(len(cloud)))
    count = 0
    print(len(cloud))
    while True:
        i = np.random.choice(indexes, 1)[0]
        p1 = cloud[i]
        n1 = cloudNormals[i]
        neighborIdx = [ii for ii in sceneTree.query(p1.reshape((1,3)), 50, distance_upper_bound = 2 * r)[1][0] if ii < len(cloud) and ii != i]
        j, k = np.random.choice(neighborIdx, 2, replace = False)

        p2, p3 = cloud[j], cloud[k]
        n2, n3 = cloudNormals[j], cloudNormals[k]
        
        if np.linalg.cond(np.column_stack((n1,n2,n3))) > 1e5:
            continue

        pose = mesh.getPose(np.column_stack((p1,p2,p3)), np.column_stack((n1, n2, n3)))
        if pose is not None:
            yield pose
        count += 1
        if count % 100 == 0:
            print(count)
    return None


if __name__ == '__main__':
    with open('Models/SCrewScene.json') as fin:
        cloud = []
        screwCloud = np.array(json.load(fin))
        for p in screwCloud:
            if not np.any(np.isnan(np.array(p))) and np.linalg.norm(np.array(p)-np.array((0,0,0.4)))< 0.3:
                cloud.append(p)
        cloud = np.array(cloud)
        fullCloud = cloud[np.random.choice(range(len(cloud)), len(cloud))]

    mesh = Mesh('Models/ToyScrew-Yellow.stl')
    mesh.compileFeatures(N = 10)


    fig = plt.figure()
    ax = fig.gca(projection = '3d')

    ax.scatter(fullCloud[:,0], fullCloud[:,1], fullCloud[:,2])
    count = 0
    for o, r in generateHypotheses(fullCloud, mesh):
        faces = (mesh.Faces @ r.T) + o
        ax.scatter(faces[:,0], faces[:,1], faces[:,2], c = 'r')
        if count > 10:
            break
        count += 1
    plt.show()



