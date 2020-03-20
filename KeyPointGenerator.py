import numpy as np
from scipy.ndimage.morphology import distance_transform_edt
from scipy.ndimage import correlate
from Graph import Graph

class KeyPointGenerator:
    SceneDistanceField = None
    SceneOrigin = None
    BinSize = None

    def __init__(self, minDistance, maxDistance, n, minNodes):
        self.MinDistance = minDistance
        self.MaxDistance = maxDistance
        self.N = n
        self.MinNodes = minNodes
    

    @staticmethod
    def distanceFieldFromCloud(cloud, binsize):
        ''' Given a cloud, make 3d binary voxel grid
            Then get the distance to the nearest 1 voxel at each grid point.
        '''
        origin = np.min(cloud, 0)
        extent = np.max(cloud, 0)
        shape = np.array((extent - origin) // binsize, dtype = np.int)
        
        distanceField = np.full(shape, 1)
        for c in cloud:
            index = np.array((c - origin) // binsize, dtype=np.int)
            if np.all(index >= 0) and np.all(index < shape):
                distanceField[tuple(index)] = 0
        distanceField = distance_transform_edt(distanceField, binsize)
        return origin, distanceField
    

    def gradient(self, distanceField):
        ''' Compute the gradient of the distance field with a 
            Simple filter.

            This can take a while for large grids. And I am not sure how to 
            extend it to sparse ones.
        '''
        fx = correlate(distanceField, np.array([
            [[-1, 0, 1]]
        ]), mode = 'nearest') / (self.BinSize * 2)
        fy = correlate(distanceField, np.array([
            [[-1],
            [0],
            [1]]
        ]), mode = 'nearest') / (self.BinSize * 2)
        fz = correlate(distanceField, np.array([
            [[-1]], [[0]], [[1]]
        ]), mode = 'nearest') / (self.BinSize * 2)
        return fx, fy, fz


    def findSkeleton(self, distanceField):
        ''' The magnitude of the gradient vector when the 
            min function is not active is always 1. So we will
            look for points that have a magnitude of < 0.8 to see where the 
            min function is operating.
        '''
        fx, fy, fz = self.gradient(distanceField)
        skeleton = np.zeros_like(distanceField)
        mask = (fx**2 + fy**2 + fz**2 < 0.8) & (distanceField > self.MinDistance) & (distanceField < self.MaxDistance)
        skeleton[mask] = distanceField[mask]
        return skeleton


    def graphIterate(self, skeleton, n, minNodes):
        g = Graph()
        g.fromMatrix(skeleton)
        g.prune(n, minNodes)
        centerNodes = g.connectedComponentCenters()
        return g.toCloud(centerNodes)


    @classmethod
    def setSceneDistanceFieldFromCloud(cls, cloud):
        KeyPointGenerator.SceneOrigin, KeyPointGenerator.SceneDistanceField = cls.distanceFieldFromCloud(cloud, KeyPointGenerator.BinSize)
    

    def keyPointsFromScene(self):
        origin, field = KeyPointGenerator.SceneOrigin, KeyPointGenerator.SceneDistanceField
        keyPoints = self.keyPointsFromField(field)
        return keyPoints + origin


    def keyPointsFromCloud(self, cloud, binsize = None):
        binsize = binsize if binsize is not None else KeyPointGenerator.BinSize

        origin, field = self.distanceFieldFromCloud(cloud, binsize)
        keyPoints = self.keyPointsFromField(field, binsize)
        return keyPoints + origin


    def keyPointsFromField(self, field, binsize = None):
        ''' Returns keypoints from a given distance field.
            Notably, this method does not take into account the origin/offset of the 
            distance field in space, so the smallest point will be at 0,0,0.
        '''
        binsize = binsize if binsize is not None else KeyPointGenerator.BinSize

        skeleton = self.findSkeleton(field)
        keyPoints, distances = self.graphIterate(skeleton, self.N, self.MinNodes)
        keyPoints = (keyPoints + 0.5) * binsize
        return keyPoints