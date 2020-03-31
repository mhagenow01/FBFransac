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
    max_iterations = 1000

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

    def getCylinder(self,p1,n1,p2,n2):
        # A cylinder is fully defined from two points with corresponding normals

        # Use the two normals to define the direction of the cylinder axis
        # a = n1 x n2
        a = np.cross(n1, n2)
        a = a/np.linalg.norm(a)

        # Project the lines from p1/p2 and the normals onto the plane ax=0
        # the intersection of these is the centerpoint of the cylinder

        # Project the first line onto the ax = 0 plane
        # we need a point as well as a direction for the line
        n1_plane = n1-a.dot(n1)*a
        p1_plane = p1-a.dot(p1)*a

        n2_plane = n2 - a.dot(n2) * a
        p2_plane = p2 - a.dot(p2) * a

        # Solve the same way as a sphere casting as an optimization to find the intersection
        # of lines in the plane (note: in this case, they should intersect!)
        X0 = np.array([0.0, 0.0])
        res = minimize(lambda x: self.distanceBetweenPointsOnLine(x, p1_plane, n1_plane, p2_plane, n2_plane), X0)
        y1_opt = res.x[0] * n1_plane + p1_plane
        y2_opt = res.x[1] * n2_plane + p2_plane
        c = (y1_opt + y2_opt) / 2

        r = (np.linalg.norm(p1_plane-c)+np.linalg.norm(p2_plane-c))/2
        return r,c,a

    def scoreCylinder(self, cloud, cloudNormals, r, c, a):
        epsilon = 0.005
        alpha = 0.2

        # Same idea as the sphere,but the points first need to be projected onto the plane!
        # For all points projected onto the cylinder plane, determine whether they are close enough (epsilon)
        # to the expected radius and have close enough normals
        cloud_on_plane = (cloud-np.multiply(np.array([np.sum(a*cloud,axis=1),np.sum(a*cloud,axis=1),np.sum(a*cloud,axis=1)]).T,a))

        consistent_pts = (np.linalg.norm(cloud_on_plane - c, axis=1) < np.abs(r) + epsilon) & (
                    np.linalg.norm(cloud_on_plane - c, axis=1) > np.abs(r) - epsilon) & (np.arccos(np.abs(
            np.sum(cloudNormals * np.divide((cloud_on_plane - c), np.linalg.norm((cloud_on_plane - c), axis=1)[:, np.newaxis]),
                   axis=1))) < alpha)

        # Get how far in the a-axis that the cylinder extends
        min_z = 0.0
        max_z = 0.0

        # If there are at least two consistent points, we can compute a bound
        if np.count_nonzero(consistent_pts)>1:
            min_z = np.min(np.sum(a * cloud[consistent_pts], axis=1))
            max_z = np.max(np.sum(a * cloud[consistent_pts], axis=1))

        return np.count_nonzero(consistent_pts), min_z, max_z

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

        # score the sphere based on points that are within an epsilon of the expected radius
        # and have normals that are moderately similar to the expected normal for a point on the sphere
        consistent_pts = (np.linalg.norm(cloud-c,axis=1) < np.abs(r)+epsilon) & (np.linalg.norm(cloud-c,axis=1) > np.abs(r)-epsilon) &\
                         (np.arccos(np.abs(np.sum(cloudNormals*np.divide((cloud-c),np.linalg.norm((cloud-c),axis=1)[:,np.newaxis]),axis=1)))<alpha)

        return np.count_nonzero(consistent_pts)

    def __init__(self):
        print("Initializing Efficient RANSAC")


    def findInCloud(self, cloud, cloudNormals):
        sceneTree = KDTree(cloud)
        indexes = list(range(len(cloud)))

        found_spheres = []
        found_cylinders = []
        r_max = 1 # 1 meter as a reasonable upper bound (largest size object) for the search
        iterations = 0

        while iterations<self.max_iterations:

            # Get the first point
            i = np.random.choice(indexes, 1)[0]
            p1 = cloud[i]
            n1 = cloudNormals[i]

            # Get other points randomly but from a reasonable neighborhood
            neighborIdx = [ii for ii in sceneTree.query(p1.reshape((1,3)), 50, distance_upper_bound = r_max)[1][0] if ii < len(cloud) and ii != i]
            j, k = np.random.choice(neighborIdx, 2, replace = False)

            p2, p3 = cloud[j], cloud[k]
            n2, n3 = cloudNormals[j], cloudNormals[k]

            # If the points don't have unique normals, we shouldn't continue since they are needed
            # to determine the geometry
            if np.linalg.cond(np.column_stack((n1,n2,n3))) > 1e5:
                continue

            #####################################################################
            # Try to fit each of the shapes given the randomly selected points  #
            #####################################################################

            # Try to fit the sphere
            r,c = self.getSphere(p1,n1,p2,n2)
            score_temp = self.scoreSphere(cloud,cloudNormals,r,c)
            # print("R: ", r, " C:", c, " %:",score_temp)

            # Scoring is proportional to the surface area of the sphere
            if score_temp > 40000*np.power(r,2):
                closest = np.inf
                for ii in range(0,len(found_spheres)):
                    if np.linalg.norm(c-found_spheres[ii][1])<closest:
                        closest = np.linalg.norm(c-found_spheres[ii][1])

                # only keep if unique (different center) from previous spheres
                if closest > 0.01:
                    found_spheres.append((r,c))

            # Try to fit the cylinder
            r, c, a = self.getCylinder(p1, n1, p2, n2)
            score_temp, min_z, max_z = self.scoreCylinder(cloud, cloudNormals, r, c, a)

            # Scoring is proportional based on surface area of the cylinder, making sure the depth
            # is at least the radius and a minimum bound to prevent fitting tiny sections
            if score_temp > 400000*np.power(r,2)*np.abs(max_z-min_z) and np.abs(max_z-min_z)>r and np.abs(max_z-min_z)>0.05 :
                closest = np.inf
                for ii in range(0, len(found_cylinders)):
                    if np.linalg.norm(c - found_cylinders[ii][1]) < closest:
                        closest = np.linalg.norm(c - found_cylinders[ii][1])

                # only keep if unique (different center) from previous cylinders
                if closest > 0.01:
                    found_cylinders.append((r, c, a, min_z, max_z))

            iterations+=1

        return found_spheres, found_cylinders
    
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
