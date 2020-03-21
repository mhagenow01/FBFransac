from Octree import Octree
import point_cloud_utils as pcu 
import numpy as np
from pykdtree.kdtree import KDTree
from Verbosifier import verbose
from KeyPointGenerator import KeyPointGenerator
from pykdtree.kdtree import KDTree
from Mesh import Mesh

class ModelFinder:
    def __init__(self):
        self.Models = []
        self.KeyPointGenerators = []
        self.Scene = None
        self.SceneKd = None
        self.MaxDistanceError = 0.001
    
    def _getKeyPointGenFromMesh(self, mesh):
        # Currently hard coded for the screw model
        return KeyPointGenerator(0.003, 0.006, 10, 500)

    @verbose()
    def set_meshes(self, meshes):
        self.Models = meshes
        for m in meshes:
            m.cacheMeshDistance()
            self.KeyPointGenerators.append(self._getKeyPointGenFromMesh(m))
    
    @verbose()
    def set_scene(self, cloud):
        self.Scene = cloud
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
            for kp in sceneKeyPoints:
                pose = self.determinePose(m, meshKeyPoints, kp.reshape((1,3)))
                if self.validatePose(m, pose):
                    instances.append((m, pose))

        return instances

    def determinePose(self, mesh, meshKp, sceneKp):
        ''' Given a mesh and a correspondence between a point in mesh space and
            a point in scene space, determine the best fit pose of the mesh.
        '''
        R = np.eye(3)
        o = sceneKp
        meshFaces = mesh.Faces - meshKp
        R, o = self.runICP(R, o, meshFaces)
        if R is None or o is None:
            return None, None

        T = np.zeros((4,4))
        T[:3,:3] = R
        T[:3, 3] = o - meshKp @ R.T
        T[ 3, 3] = 1
        return T[:3,:3], T[:3,3]

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
        print(outliers, inliers)
        return True
        if outliers > 0:
            return False
        if inliers < 60:
            return False
        return True

    def runICP(self, R, o, mesh):
        ''' Given a current pose (R,  o) for a mesh, use ICP to iterate
        and find a better pose that aligns the closest points
        '''
        # Parameters for ICP
        max_iterations = 1 # max iterations for a mesh to preserve performance
        tolerance = 0.025 # when to stop ICP -> cumulative error
        distance_threshold = 0.02 # 2 cm away for closest point

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

            # TODO: Add weights to the pairs of points (SKIP FOR NOW)
            # Can be tuned based on things like normals, etc.

            centroid_s = np.mean(s_vals, 0)
            centroid_m = np.mean(m_vals, 0)
            s_vals -= centroid_s
            m_vals -= centroid_m
            S = s_vals.T @ m_vals
            U, sigma, Vh = np.linalg.svd(S)
            V = np.transpose(Vh)

            # Rotation using SVD
            R_new = V @ U.T
            t = (centroid_s - centroid_m @ R_new.T)

            # Update poses - NOTE: translation and rotation
            # are with respect to the previous values for rotation and translation
            R, o = R_new @ R, o @ R_new.T + t

            # Compute the summed error E(R,t) to determine whether another iteration should be done
            face_points_temp = mesh @ R.T + o
            distances_temp, closest_indices_temp = self.SceneKd.query(face_points_temp, 1)
            closest_points_temp = self.Scene[closest_indices_temp]
            error = np.sum(np.linalg.norm(closest_points_temp-face_points_temp,axis=1))

            number_iterations += 1

        print(number_iterations)
        return R, o
