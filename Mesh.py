import trimesh
from trimesh.proximity import ProximityQuery
import pickle
import os
import time
import numpy as np
from scipy.spatial.transform import Rotation
from pykdtree.kdtree import KDTree
import itertools
from Verbosifier import verbose
import rtree
import functools
import progressbar

FEATURE_CACHE_DIR = os.path.join(os.curdir,'FeatureCache')

class Mesh:
    def __init__(self, meshFile):
        self.trimesh = trimesh.load_mesh(meshFile)
        self.Radius = self.trimesh.bounding_sphere._data['radius']
        self.Features = None
        self.determineFacesAndNormals()
        self.FeatureTree = None
        

    @verbose(1000)
    def getPose(self, P, N):
        #TODO: This only works right now if you sample the points in the "right" order.
        # I.E. the same order that the faces are stored in the feature.
        # Could probably order the points and features somehow consistently.

        pointFeature = self._computePointFeature(P, N)
        faceSets = self.FeatureTree.intersection(pointFeature, objects = True)
        for faceSet in faceSets:
            for fset in itertools.permutations(faceSet.object, 3):
                _lfset = list(fset)
                F = self.Faces[_lfset]
                FN = self.Normals[_lfset]
                got, *pose = self.getPoseFromCorrespondence(P, N, F, FN)
                if got:
                    yield pose
        return None

    
    @verbose(1000)
    def getPoseFromCorrespondence(self, P, N, F, FN):
        R = N @ np.linalg.inv(FN)
        R = Rotation.match_vectors(N.T, FN.T)[0].as_dcm()
        u, s, vt = np.linalg.svd(R)
        if np.any(np.abs(s ** 2 - 1) > 0.01):
            print('Cant rotate')
            return False, None, None
        #TODO:I am pretty sure these two lines are equivalent
        b = np.sum(P * N, axis = 0) - np.sum(F * FN, axis = 0)
        origin = np.linalg.solve(N.T, b)


        relative = (P.T - origin.reshape((1, 3))) @ R
        # This is slightly better than the lower option
        if np.any(np.linalg.norm(relative, axis = 1) >  1.25 * self.Radius):
            return False, None, None
        #TODO: This is horribly inefficient for just checking 3 points.
        # Can be optimized by looking at whether or not the point lies inside of the
        # face that it is supposed to.
        distance = ProximityQuery(self.trimesh).signed_distance(relative)
        #TODO: This tolerance should be configurable.
        if np.any(np.abs(distance) > 0.002):
            #print('Rejected by distance')
            #print(distance)
            return False, None, None
            
        return True, origin, R


    @verbose()
    def compileFeatures(self, **kwargs):
        """Generate a set of features for a mesh.
        
        :param N: the number of features to generate
        :type N: int
        """                                                                        
        if not os.path.isdir(FEATURE_CACHE_DIR):
            os.mkdir(FEATURE_CACHE_DIR)
        cacheFile = self.getCacheName(kwargs)


        if os.path.isfile(cacheFile):
            print(f'Loading features from {cacheFile}')
            with open(cacheFile, 'br') as fin:
                self.Features = pickle.load(fin)
        else:
            print('No feature cache found. Compiling features.')
            s = time.time()
            self._compileFeatures(**kwargs)
            self._cacheFeatures(**kwargs)
            print(f'Generated {len(self.Features)} features in {time.time() - s:.4}s.')
        self._createFeatureTree()
        return
            

    @verbose()
    def _compileFeatures(self, **kwargs):
        N = kwargs.get('N')
        n = len(self.Faces)
        self.Features = []
        adjacencySet = set(tuple(a) for a in self.trimesh.face_adjacency)
        bar = progressbar.ProgressBar(n * (n - 1) * (n - 2) / 6, widgets=[progressbar.Bar('=', '[', ']')])
        count = 0
        bar.start()
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    count += 1
                    ijk = np.array((i,j,k))
                    ns = self.Normals[ijk]
                    ss = self.Sizes[ijk]
                    # if ((i,j) not in adjacencySet and (i,k) not in adjacencySet) or \
                    #     ((i,j) not in adjacencySet and (j,k) not in adjacencySet) or \
                    #     ((i,k) not in adjacencySet and (j,k) not in adjacencySet):
                    #     continue
                    if np.linalg.cond(ns) > 1e5:
                        continue

                    self.Features.append((np.sum(ss, keepdims= False), ijk))
                    bar.update(count)
        bar.finish()

        self.Features = sorted(self.Features, reverse = True, key = lambda x : x[0])[:N]
        return
    

    @verbose()
    def _cacheFeatures(self, **kwargs):
        name = self.getCacheName(kwargs)
        print(f'Writing to feature cache {name}')
        with open(name, 'bw') as fout:
            pickle.dump(self.Features, fout)
        return

    @verbose()
    def _createFeatureTree(self):
        featureVectors = []

        for i, (score, faceSet) in enumerate(self.Features):
            normals = self.Normals[faceSet]
            # Format for rtree is x_low, y_low, z_low... , x_high, y_high, z_high...
            innerProductTolerance = 0.01
            featureVector = [
                abs(normals[0].dot(normals[1])) - innerProductTolerance,
                abs(normals[0].dot(normals[2])) - innerProductTolerance,
                abs(normals[1].dot(normals[2])) - innerProductTolerance
            ]
            lowDistance, highDistance = list(zip(*list(self.distanceRange(f1, f2) for f1,f2 in itertools.combinations(faceSet, 2))))
            featureVector.extend(lowDistance)
            featureVector.extend([
                abs(normals[0].dot(normals[1])) + innerProductTolerance,
                abs(normals[0].dot(normals[2])) + innerProductTolerance,
                abs(normals[1].dot(normals[2])) + innerProductTolerance
                
            ])
            featureVector.extend(highDistance)


            # Rtree needs an ID for the rectangle (i), the bounds (featureVector), and can accept something to 
            # associate with that rectangle (the face set in this case)
            featureVectors.append((i, featureVector, faceSet))

        # Just grab the first one to check what the dimension is
        prop = rtree.index.Property()
        prop.dimension = len(featureVectors[0][1]) // 2
        self.FeatureTree = rtree.index.Index(featureVectors, properties = prop, objects = 'raw')
        return

    def _computePointFeature(self, P, N):
        featureVector = []
        ntn = N.T @ N
        normals = abs(np.array([ntn[0,1], ntn[0,2], ntn[1, 2]]))
        distances = []
        for i,j in itertools.combinations(range(3), 2):
            distances.append(np.linalg.norm(P[:,i] - P[:,j]))
        featureVector.extend(normals)
        featureVector.extend(distances)
        featureVector.extend(np.array(featureVector) + 0.000001)
        return featureVector


    def getCacheName(self, kwargs):
        name = '-'.join(f'{k}_{v}' for k,v in kwargs.items())
        version = '0.3.1'
        return os.path.join(FEATURE_CACHE_DIR,f'{name}.{version}.ftr')


    def determineFacesAndNormals(self):
        """Generates arrays of the positions of the faces, the normals, 
            and the sizes of each face.
        """        
        mesh = self.trimesh
        self.Faces = np.zeros((len(mesh.faces), 3))
        self.Normals = np.zeros((len(mesh.faces), 3))
        self.Sizes = np.zeros((len(mesh.faces), 1))
        for i,v_ind in enumerate(mesh.faces):
            self.Normals[i] = np.cross(mesh.vertices[v_ind[1]] - mesh.vertices[v_ind[0]], mesh.vertices[v_ind[2]] - mesh.vertices[v_ind[0]])
            self.Sizes[i] = abs(np.linalg.norm(self.Normals[i])) / 2
            self.Normals[i] /= np.linalg.norm(self.Normals[i])
            self.Faces[i] = np.mean(mesh.vertices[v_ind], axis = 0)
        return

    def distanceRange(self, f1, f2):
        distances = np.array(list( map(self.vertDistance, itertools.product(self.trimesh.faces[f1], self.trimesh.faces[f2])) ))
        return np.min(distances), np.max(distances)

    def vertDistance(self, v12):
        v1, v2 = v12
        return np.linalg.norm(self.trimesh.vertices[v1] - self.trimesh.vertices[v2])
