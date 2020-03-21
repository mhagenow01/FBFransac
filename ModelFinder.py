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
        self.MaxDistanceError = 0.004
    
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
        # TODO: This currently has ICP as an external step, so we can't pass back a real pose
        # or do validation. That should be integrated into this.
        # When ICP is in here, we should change what this returns.
        instances = []
        for m, kg in zip(self.Models, self.KeyPointGenerators):
            # First generate hypotheses of scene model correspondences.
            sceneKeyPoints = kg.keyPointsFromScene()
            # TODO: This should be cached on a per-mesh basis.
            meshKeyPoints = kg.keyPointsFromField(m.DistanceCache) + m.BoundingBoxOrigin
            # TODO: Allow for potentially multiple keypoints per mesh?
            meshKeyPoints = meshKeyPoints[0:1,:]
            for kp in sceneKeyPoints:
                pose = self.determinePose(m, meshKeyPoints, kp)
                if self.validatePose(m, pose):
                    instances.append((m, kp, meshKeyPoints))

        return instances

    def determinePose(self, mesh, meshKp, sceneKp):
        ''' Given a mesh and a correspondence between a point in mesh space and
            a point in scene space, determine the best fit pose of the mesh.
        '''
        R = np.eye(3)
        o = sceneKp
        meshFaces = mesh.Faces - meshKp
        R,o = self.runICP(R,o,meshFaces)

        t = np.zeros((4,4))
        t[:3,:3] = R
        t[:3, 3] = o
        t[ 3, 3] = 1
        kpShift = np.zeros((4,4))
        kpShift[:3,:3] = np.eye(3)
        kpShift[:3, 3] = -meshKp
        kpShift[ 3, 3] = 1
        T = t @ kpShift
        return T[:3,:3], T[:3,3]

    def validatePose(self, mesh : Mesh, pose):
        ''' Given a mesh, pose, and representation of the scene (self.SceneKd), figure out how
            good the pose is at describing the scene.

            Then return True if its good enough, and False otherwise.
        '''
        R, o = pose
        nearbyDistances, nearbyPoints_ind = self.SceneKd.query(o.reshape((1,3)), k = 300, distance_upper_bound = mesh.Radius)
        nearbyPoints_ind = np.array(nearbyPoints_ind)
        nearbyPoints = self.Scene[nearbyPoints_ind[nearbyPoints_ind < len(self.Scene)]]

        nearbyPoints = (nearbyPoints - o) @ R
        distanceToMesh = mesh.distanceQuery(nearbyPoints)
        outliers = np.sum(distanceToMesh > self.MaxDistanceError)
        inliers = np.sum(np.abs(distanceToMesh) <= self.MaxDistanceError)
        if outliers > 0:
            return False
        if inliers < 60:
            return False
        return True

    def runICP(self, R, o, mesh):
        ''' Given a current pose (R + o) for a mesh, use ICP to iterate
        and find a better pose that aligns the closest points
        '''

        # Parameters for ICP
        max_iterations = 20 # max iterations for a mesh to preserve performance
        tolerance = 0.25 # when to stop ICP -> cumulative error
        distance_threshold = 0.02 # 2 mm away for closest point


        # starting value for exit conditions
        number_iterations = 0
        error = np.inf

        while (error > tolerance) and (number_iterations < max_iterations):
            # Compute the nearest point in the point cloud for each point in the model

            face_points = mesh @ R + o.reshape((1,3))
            distances, closest_indices = self.SceneKd.query(face_points, 1)
            closest_points = self.Scene[closest_indices]

            closeEnough = distances < distance_threshold
            s_vals = closest_points[closeEnough]
            m_vals = face_points[closeEnough]

            # TODO: Add weights to the pairs of points (SKIP FOR NOW)
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
            R_new = np.matmul(V,np.transpose(U))
            t = (centroid_m.reshape((3,1))- R_new @ centroid_s.reshape((3,1)))

            # REMOVE
            # t = np.zeros((3,1))
            # print ("R in ICP:", R)
            # print ("T in ICP:", t)

            # Update poses - NOTE: translation and rotation
            # are with respect to the previous values for rotation and translation
            (R,o) = ((R_new.T @ R.T,o.reshape((3,))-t.reshape((3,))))

            # Compute the summed error E(R,t) to determine whether another iteration should be done
            face_points_temp = mesh @ R + o.reshape((1, 3))
            distances_temp, closest_indices_temp = self.SceneKd.query(face_points_temp, 1)
            closest_points_temp = self.Scene[closest_indices_temp]
            error = np.sum(np.linalg.norm(closest_points_temp-face_points_temp,axis=1))

            # print("ITERATION: ",number_iterations," Error: ",error)

            number_iterations+=1


        return R,o
