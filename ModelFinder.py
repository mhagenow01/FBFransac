from Octree import Octree
import point_cloud_utils as pcu 
import numpy as np
from pykdtree.kdtree import KDTree
from Verbosifier import verbose
from KeyPointGenerator import KeyPointGenerator
from pykdtree.kdtree import KDTree
from Mesh import Mesh
from scipy.stats import special_ortho_group
import random


class ModelFinder:
    def __init__(self):
        self.Models = []
        self.KeyPointGenerators = []
        self.Scene = None
        self.SceneNormals = None
        self.SceneKd = None
        self.MaxDistanceError = 0.001
    
    def _getKeyPointGenFromMesh(self, mesh):
        # Currently hard coded for the screw model
        return KeyPointGenerator(0.003, 0.006, 10, 1000)

    @verbose()
    def set_meshes(self, meshes):
        self.Models = meshes
        for m in meshes:
            m.cacheMeshDistance()
            self.KeyPointGenerators.append(self._getKeyPointGenFromMesh(m))
    
    @verbose()
    def set_scene(self, cloud):
        self.Scene = cloud
        self.SceneNormals = pcu.estimate_normals(cloud,10,3)
        self.SceneKd = KDTree(cloud)
        KeyPointGenerator.setSceneDistanceFieldFromCloud(cloud)


    def set_resolution(self, res):
        KeyPointGenerator.BinSize = res
    

    @verbose()
    def findInstances(self):
        ''' Using the defined meshes, keypoint generators, and scene cloud,
            find all of the potential mesh positions. 

            Returns: (the mesh, the scene keypoint, the corresponding mesh keypoint)
        '''
        instances = []
        for m, kg in zip(self.Models, self.KeyPointGenerators):
            # First generate hypotheses of scene model correspondences.
            sceneKeyPoints = kg.keyPointsFromScene()
            # TODO: This should be cached on a per-mesh basis.
            meshKeyPoints = kg.keyPointsFromField(m.DistanceCache) + m.BoundingBoxOrigin
            # TODO: Allow for potentially multiple keypoints per mesh?
            meshKeyPoints = meshKeyPoints[0:1,:]
            iter = 0
            for kp in sceneKeyPoints:
                if 1: # MH: temporary lines just so I can make it only spit out one screw during unit-testing
                    pose = self.determinePose(m, meshKeyPoints, kp.reshape((1,3)))
                    if self.validatePose(m, pose):
                        instances.append((m, pose))
                iter+=1
        return instances

    def determinePose(self, mesh, meshKp, sceneKp):
        ''' Given a mesh and a correspondence between a point in mesh space and
            a point in scene space, determine the best fit pose of the mesh.
        '''
        R = np.eye(3)
        o = sceneKp
        meshFaces = mesh.Faces - meshKp
        R, o, error = self.ICPrandomRestarts(R, o, meshFaces, mesh.Normals)
        if R is None or o is None:
            return None, None

        return R, o - meshKp @ R.T

    def validatePose(self, mesh : Mesh, pose):
        ''' Given a mesh, pose, and representation of the scene (self.SceneKd), figure out how
            good the pose is at describing the scene.

            Then return True if its good enough, and False otherwise.
        '''
        R, o = pose
        if R is None or o is None:
            return False
        nearbyDistances, nearbyPoints_ind = self.SceneKd.query(o.reshape((1,3)), k = 300, distance_upper_bound = mesh.Radius)
        nearbyPoints_ind = np.array(nearbyPoints_ind)
        nearbyPoints = self.Scene[nearbyPoints_ind[nearbyPoints_ind < len(self.Scene)]]

        nearbyPoints = (nearbyPoints - o) @ R
        distanceToMesh = mesh.distanceQuery(nearbyPoints)
        outliers = np.sum(distanceToMesh > self.MaxDistanceError)
        inliers = np.sum(np.abs(distanceToMesh) <= self.MaxDistanceError)
        # print(outliers, inliers)
        return True
        if outliers > 0:
            return False
        if inliers < 60:
            return False
        return True


    def ICPrandomRestarts(self,R,o,mesh, meshNormals):
        number_restarts = 500
        best_error = np.inf

        for ii in range(0,number_restarts):
            R = special_ortho_group.rvs(3) # random restart for R_initial

            o_pert = 0.1*np.array([random.random()-0.5, random.random()-0.5,random.random()-0.5]).reshape((1,3))
            # print(o)
            # print(o_pert)
            # print(o+o_pert)

            R_temp, o_temp, error = self.runICP(R,o+o_pert,mesh,meshNormals)

            if error<best_error:
                best_error = error
                best_R, best_o = R_temp, o_temp

        return best_R, best_o, best_error

    def runICP(self, R, o, mesh, meshNormals):
        ''' Given a current pose (R,  o) for a mesh, use ICP to iterate
        and find a better pose that aligns the closest points
        '''
        # Parameters for ICP
        max_iterations = 50 # max iterations for a mesh to preserve performance
        tolerance = 0.001*len(mesh) # when to stop ICP -> cumulative error
        distance_threshold = 0.1 # 2 cm away for closest point

        # starting value for exit conditions
        number_iterations = 0
        error = np.inf

        while (error > tolerance) and (number_iterations < max_iterations):
            # Compute the nearest point in the point cloud for each point in the model

            face_points = mesh @ R.T + o
            distances, closest_indices = self.SceneKd.query(face_points, 1)
            closest_points = self.Scene[closest_indices]

            closeEnough = distances < distance_threshold
            s_vals = closest_points[closeEnough]
            m_vals = face_points[closeEnough]
            if len(s_vals) < 3:
                return None, None


            #########################################
            # Ability to add weights to the points  #
            #########################################

            # All ones is no-weighting
            weights = np.ones((len(s_vals),))

            # # weights based on distance
            # weights = 1.0 - (np.abs(distances[closeEnough])/distance_threshold)

            # # weights based on normals
            # weights = np.abs(np.sum(self.SceneNormals[closest_indices][closeEnough]*(meshNormals[closeEnough] @ R.T),axis=1))


            # print("weights:",weights)

            weights_matrix = np.diag(weights)

            centroid_s = np.divide(s_vals.T @ weights ,np.sum(weights))
            centroid_m = np.divide(m_vals.T @ weights ,np.sum(weights))

            s_vals -= centroid_s
            m_vals -= centroid_m
            S = m_vals.T @ weights_matrix @ s_vals
            U, sigma, Vh = np.linalg.svd(S)
            V = np.transpose(Vh)

            # Rotation using SVD
            R_new = V @ U.T
            t = (centroid_s - centroid_m @ R_new.T)

            # print("ICP R:", R_new, " T: ",t)
            # print("Centroid S:",centroid_s, "Centroid M:", centroid_m)
            # print("LEN: ", len(s_vals))

            # Update poses - NOTE: translation and rotation
            # are with respect to the previous values for rotation and translation
            # print("OLD R:", R, " T:",o)
            R, o =   R_new @ R, o @ R_new.T + t
            # print("NEW R:", R, " T:", o)

            # Compute the summed error E(R,t) to determine whether another iteration should be done
            face_points_temp = mesh @ R.T + o
            distances_temp, closest_indices_temp = self.SceneKd.query(face_points_temp, 1)
            closest_points_temp = self.Scene[closest_indices_temp]
            error = np.sum(np.linalg.norm(closest_points_temp-face_points_temp,axis=1))
            # print("ERROR ", error)
            number_iterations += 1

        print(number_iterations)
        return R, o, error
