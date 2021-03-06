import point_cloud_utils as pcu 
import numpy as np
from pykdtree.kdtree import KDTree

class ModelFinder:
    def __init__(self, model):
        self.Model = model
    
    #Don't try to verbose a generator.
    def findInCloud(self, cloud, cloudNormals):
        sceneTree = KDTree(cloud)
        indexes = list(range(len(cloud)))

        r = self.Model.Radius
        iterations = 0
        while True:
            iterations += 1
            if iterations % 100 == 0:
                print(iterations)
            i = np.random.choice(indexes, 1)[0]
            p1 = cloud[i]
            n1 = cloudNormals[i]
            neighborIdx = [ii for ii in sceneTree.query(p1.reshape((1,3)), 50, distance_upper_bound = 2 * r)[1][0] if ii < len(cloud) and ii != i]
            j, k = np.random.choice(neighborIdx, 2, replace = False)

            p2, p3 = cloud[j], cloud[k]
            n2, n3 = cloudNormals[j], cloudNormals[k]
            
            if np.linalg.cond(np.column_stack((n1,n2,n3))) > 1e5:
                continue
            
            for pose in self.Model.getPose(np.column_stack((p1,p2,p3)), np.column_stack((n1, n2, n3))):
                yield pose

        return None
    
    @staticmethod
    def voxelFilter(cloud, size = 0.01):
        min = np.min(cloud, axis = 0)
        max = np.max(cloud, axis = 0)
        nBins = np.array(np.ceil((max - min) / size), dtype = np.int)
        grid = np.full(nBins, -1, dtype = np.int)
        nPoints = 0
        for i, p in enumerate(cloud):
            index = np.array(np.floor((p - min) / size), dtype = np.int)
            t_index = tuple(index)
            j = grid[t_index]
            if j == -1:
                grid[t_index] = i
                nPoints += 1
            else:
                gridPos = min + (index / nBins) * (max - min)
                if np.linalg.norm(p - gridPos) < np.linalg.norm(cloud[grid[t_index]] - gridPos):
                    grid[t_index] = i
        chosenPoints = np.zeros(nPoints, dtype = np.int)
        i = 0
        for pointIndex in grid.flatten():
            if pointIndex > -1:
                chosenPoints[i] = pointIndex
                i += 1
        return chosenPoints

    @staticmethod
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

    
    @staticmethod
    def meanPlanarCloudSampling(cloud, cloudNormals, radius = 0.1, normalThreshold = 0.1, coplanarThreshold = 0.01):
        print('Sampling cloud!')
        sampledPoints = []
        sampledNormals = []
        count = []
        for p, n in zip(cloud, cloudNormals):
            represented = False
            for i, (p2, n2, c) in enumerate(zip(sampledPoints, sampledNormals, count)):
                if abs(n.dot(n2)) < 1 - normalThreshold:
                    continue
                p_r = p - p2
                if abs(p_r.dot(n2)) > coplanarThreshold:
                    continue
                p_n2 = p_r - p_r.dot(n2) * n2
                if np.linalg.norm(p_r) > radius:
                    continue
                represented = True
                sampledPoints[i] = (p2 * c + p) / (c + 1)
                if n.dot(n2) < 0:
                    n = -n
                sampledNormals[i] = (n2 * c + n) / (c + 1)
                count[i] += 1
                break
                
            if not represented:
                sampledPoints.append(p)
                sampledNormals.append(n)
                count.append(1)

        return np.array(sampledPoints), np.array(sampledNormals)

