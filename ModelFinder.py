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
        self.MaxDistanceError = 0.005
    
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
        # TODO: Do some ICP. Update R and o

        t = np.zeros((4,4))
        t[:3,:3] = R
        t[:3, 3] = o
        t[ 3, 3] = 1
        kpShift = np.zeros((4,4))
        kpShift[:3,:3] = np.eye(3)
        kpShift[:3, 3] = -meshKp
        kpShift[ 3, 3] = 1
        T = t @ kpShift
        print(t)
        print(T)
        return T[:3,:3], T[:3,3]

    def validatePose(self, mesh : Mesh, pose):
        ''' Given a mesh, pose, and representation of the scene (self.SceneKd), figure out how
            good the pose is at describing the scene.

            Then return True if its good enough, and False otherwise.
        '''
        R, o = pose
        print(o)
        nearbyDistances, nearbyPoints_ind = self.SceneKd.query(o.reshape((1,3)), k = 300, distance_upper_bound = mesh.Radius)
        nearbyPoints_ind = np.array(nearbyPoints_ind)
        nearbyPoints = self.Scene[nearbyPoints_ind[nearbyPoints_ind < len(self.Scene)]]

        nearbyPoints = (nearbyPoints - o) @ R
        distanceToMesh = mesh.distanceQuery(nearbyPoints)
        print(distanceToMesh)
        outliers = np.sum(distanceToMesh > self.MaxDistanceError)
        inliers = np.sum(np.abs(distanceToMesh) <= self.MaxDistanceError)
        print(outliers)
        print(inliers)
        if outliers > 0:
            return False
        if inliers < 60:
            return False
        return True
