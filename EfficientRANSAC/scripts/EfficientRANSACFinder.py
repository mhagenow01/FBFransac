import numpy as np
from pykdtree.kdtree import KDTree
from scipy.optimize import minimize
import math
import time

################################################################
#  Implements the Efficient RANSAC Shape Primitive Algorithm   #
# described in http://www.hinkali.com/Education/PointCloud.pdf #
################################################################

class EfficientRANSACFinder:

    ###########################################
    # Class Members                           #
    ###########################################


    ###########################################
    # Utility Functions                       #
    ###########################################

    def distanceBetweenPointsOnLine(self,x,p1,n1,p2,n2):
        y1 = x[0]*n1+p1
        y2 = x[1]*n2+p2
        return np.linalg.norm(y1-y2)

    ###########################################
    # Models for each of the shape primitives #
    ###########################################

    def getCylinder(self):
        print("Cylinder")

    def scoreCylinder(self):
        print("Cylinder")

    def getSphere(self,p1,n1,p2,n2):
        # A sphere is fully defined from two points with corresponding normals

        # Midpoint of the shortest line segment between the two lines given from p1/p2 and the normals
        # to define the centerpoint of the sphere

        # Cast this problem as an optimization two find the two points on the line creating the segment
        # y1 = t1*n1+p1
        # y2 = t2*n2+p2
        # trying to optimize t1 and t2
        X0 = np.array([0.0, 0.0])
        res = minimize(lambda x: self.distanceBetweenPointsOnLine(x,p1,n1,p2,n2),X0)
        y1_opt = res.x[0]*n1+p1
        y2_opt = res.x[1]*n2+p2
        c = (y1_opt+y2_opt)/2

        # The radius can be calculated using the center and the original points
        r = (np.linalg.norm(p1-c)+np.linalg.norm(p2-c))/2
        return r,c

    def scoreSphere(self,cloud,cloudNormals,r,c):
        epsilon = 0.005
        alpha = 0.2
        consistent_pts = (np.linalg.norm(cloud-c,axis=1) < np.abs(r)+epsilon) & (np.linalg.norm(cloud-c,axis=1) > np.abs(r)-epsilon) & (np.arccos(np.abs(np.sum(cloudNormals*np.divide((cloud-c),np.linalg.norm((cloud-c),axis=1)[:,np.newaxis]),axis=1)))<alpha)

        # TODO: Needs largest connected component

        return np.count_nonzero(consistent_pts)

    def __init__(self):
        print("Initializing Efficient RANSAC")


    def findInCloud(self, cloud, cloudNormals):
        sceneTree = KDTree(cloud)
        indexes = list(range(len(cloud)))


        found_spheres = []
        r_max = 1 # TODO: come back to this
        iterations = 0
        while iterations<1000:
            i = np.random.choice(indexes, 1)[0]
            p1 = cloud[i]
            n1 = cloudNormals[i]
            neighborIdx = [ii for ii in sceneTree.query(p1.reshape((1,3)), 50, distance_upper_bound = r_max)[1][0] if ii < len(cloud) and ii != i]
            j, k = np.random.choice(neighborIdx, 2, replace = False)

            p2, p3 = cloud[j], cloud[k]
            n2, n3 = cloudNormals[j], cloudNormals[k]
            
            if np.linalg.cond(np.column_stack((n1,n2,n3))) > 1e5:
                continue

            r,c = self.getSphere(p1,n1,p2,n2)
            score_temp = self.scoreSphere(cloud,cloudNormals,r,c)
            # print("R: ", r, " C:", c, " %:",score_temp)
            if score_temp > 40000*np.power(r,2):
                found_spheres.append((r,c))
            iterations+=1

        return found_spheres
    
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
