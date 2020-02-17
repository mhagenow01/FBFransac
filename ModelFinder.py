from Octree import Octree
import point_cloud_utils as pcu 
import numpy as np
from pykdtree.kdtree import KDTree

class ModelFinder:
    def __init__(self, model):
        self.Model = model
    
    def findInCloud(self, cloud):
        cloudNormals = pcu.estimate_normals(cloud, 5)
        cloud, cloudNormals = self.planarCloudSampling(cloud, cloudNormals, 0.1, 0.3, 0.001)
        sceneTree = KDTree(cloud)
        indexes = list(range(len(cloud)))

        r = self.Model.Radius
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

            pose = self.Model.getPose(np.column_stack((p1,p2,p3)), np.column_stack((n1, n2, n3)))
            if pose is not None:
                yield pose

        return None
    
    def planarCloudSampling(self, cloud, cloudNormals, radius = 0.1, normalThreshold = 0.1, coplanarThreshold = 0.01):
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
