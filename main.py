from Octree import Octree
import trimesh
import numpy as np
import matplotlib.pyplot as plt
import itertools
import json
import point_cloud_utils as pcu
from pykdtree.kdtree import KDTree
from trimesh.proximity import ProximityQuery
from scipy.spatial.transform import Rotation
from mpl_toolkits.mplot3d import Axes3D


def facesAndNormals(mesh):
    positions = np.zeros((len(mesh.faces), 3))
    normals = np.zeros((len(mesh.faces), 3))
    sizes = np.zeros((len(mesh.faces), 1))
    for i,v_ind in enumerate(mesh.faces):
        normals[i] = np.cross(mesh.vertices[v_ind[1]] - mesh.vertices[v_ind[0]], mesh.vertices[v_ind[2]] - mesh.vertices[v_ind[0]])
        sizes[i] = abs(np.linalg.norm(normals[i])) / 2
        normals[i] /= np.linalg.norm(normals[i])
        positions[i] = np.mean(mesh.vertices[v_ind], axis = 0)

    return positions, normals, sizes


def getMeshFeatures(mesh):
    print('Finding features!')
    faces, normals, sizes = facesAndNormals(mesh)
    n = len(faces)
    features = []
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                #fs = faces[ijk]
                ijk = (i,j,k)
                ns = normals[np.array(ijk)]
                ss = sizes[np.array(ijk)]

                if np.linalg.cond(ns) > 1e5:
                    continue
                features.append((sum(ss), ijk))
    features = sorted(features, reverse = True)
    return [np.array(f[1]) for f in features[:10]], faces, normals

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
 

def getHypothesis(p1, p2, p3, n1, n2, n3, f1, f2, f3, fn1, fn2, fn3, mesh):
    B = np.column_stack((fn1, fn2, fn3))
    B_prime = np.column_stack((n1, n2, n3))
    R = B_prime @ np.linalg.inv(B)
    R = Rotation.match_vectors(B_prime.T, B.T)[0].as_dcm()
    u, s, vt = np.linalg.svd(R)
    if np.any(np.abs(s ** 2 - 1) > 0.01):
        #print('Cant rotate')
        return False, None, None
    b = np.array((p1.dot(n1) - (R@f1).dot(n1), p2.dot(n2) - (R@f2).dot(n2), p3.dot(n3) - (R@f3).dot(n3)))
    origin = np.linalg.solve(B_prime.T, b)
    if np.any(np.abs(origin)) > 100:
        return False, None, None
    relative = (np.vstack((p1,p2,p3)) - origin.reshape((1, 3))) @ R

    try:
        distance = ProximityQuery(mesh).signed_distance(relative)
    except:
        print(relative)
    if np.any(np.abs(distance) > 0.002):
        #print('Rejected by distance')
        return False, None, None
    return True, origin, R


def generateHypotheses(cloud, mesh):
    print('Finding Hypotheses!')
    cloudNormals = pcu.estimate_normals(fullCloud, 5)
    cloud, cloudNormals = planarCloudSampling(fullCloud, cloudNormals, 0.1, 0.3, 0.001)
    sceneTree = KDTree(cloud)
    r = mesh.bounding_sphere._data['radius']
    features, faces, normals = getMeshFeatures(mesh)

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

        for feature in features:
            f1,f2,f3 = faces[feature]
            fn1, fn2, fn3 = normals[feature]

            got, *pose = getHypothesis(p1, p2, p3, n1, n2, n3, f1, f2, f3, fn1, fn2, fn3, mesh)
            if got:
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
    mesh = trimesh.load_mesh('Models/ToyScrew-Yellow.stl')


    fig = plt.figure()
    ax = fig.gca(projection = '3d')

    ax.scatter(fullCloud[:,0], fullCloud[:,1], fullCloud[:,2])
    faces, *_ = facesAndNormals(mesh)
    count = 0
    for o, r in generateHypotheses(fullCloud, mesh):
        theseFaces = (faces @ r.T) + o
        ax.scatter(theseFaces[:,0], theseFaces[:,1], theseFaces[:,2], c = 'r')
        if count > 100:
            break
        count += 1
    plt.show()



