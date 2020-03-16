from Octree import Octree
import point_cloud_utils as pcu 
import numpy as np
from pykdtree.kdtree import KDTree
from Verbosifier import verbose
from KeyPointGenerator import KeyPointGenerator

class ModelFinder:
    def __init__(self):
        self.Models = []
        self.KeyPointGenerators = []
        self.Scene = None
    
    def _getKeyPointGenFromMesh(self, mesh):
        # Currently hard coded for the screw model
        return KeyPointGenerator(0.003, 0.006, 10, 500)

    @verbose()
    def set_meshes(self, meshes):
        self.Models = meshes
        for m in meshes:
            m.cacheMeshDistance(KeyPointGenerator.BinSize)
            self.KeyPointGenerators.append(self._getKeyPointGenFromMesh(m))
    
    @verbose()
    def set_scene(self, cloud):
        self.Scene = cloud
        KeyPointGenerator.setSceneDistanceFieldFromCloud(cloud)


    def set_resolution(self, res):
        KeyPointGenerator.BinSize = res
    

    @verbose()
    def findInstances(self):
        instances = []
        for m,kg in zip(self.Models, self.KeyPointGenerators):
            # First generate hypotheses of scene model correspondences.
            sceneKeyPoints = kg.keyPointsFromScene()
            meshKeyPoints = kg.keyPointsFromField(m.DistanceCache) + m.BoundingBoxOrigin
            # TODO: Allow for potentially multiple keypoints per mesh?
            meshKeyPoints = meshKeyPoints[0:1,:]
            for kp in sceneKeyPoints:
                pose = self.determinePose(m, meshKeyPoints, kp)
                if self.validatePose(m, pose, self.Scene):
                    instances.append((m, pose))

        return instances

    def determinePose(self, mesh, meshKp, sceneKp):
        ''' Given a mesh and a correspondence between a point in mesh space and
            a point in scene space, determine the best fit pose of the mesh.
        '''
        # TODO: Actually do something here, rather than just picking a nearby point.
        # This is where some ICP would go.
        return sceneKp, np.eye(3)

    def validatePose(self, mesh, pose, scene):
        ''' Given a mesh, pose, and representation of the scene, figure out how
            good the pose is at describing the scene.

            Then return True if its good enough, and False otherwise.
        '''
        # TODO: Actually do something, like compare the distance fields 
        # of the scene and the mesh or something.

        return True
