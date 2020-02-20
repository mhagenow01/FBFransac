import trimesh
from trimesh.proximity import ProximityQuery
import pickle
import os
import time
import numpy as np
from scipy.spatial.transform import Rotation

FEATURE_CACHE_DIR = os.path.join(os.curdir,'FeatureCache')

class Mesh:
    def __init__(self, meshFile):
        self.trimesh = trimesh.load_mesh(meshFile)
        self.Radius = self.trimesh.bounding_sphere._data['radius']
        self.Features = None
        self.determineFacesAndNormals()
        

    def getPose(self, P, N):
        #TODO: This only works right now if you sample the points in the "right" order.
        # I.E. the same order that the faces are stored in the feature.
        # Could probably order the points and features somehow consistently.
        for score, feature in self.Features:
            F = self.Faces[feature]
            FN = self.Normals[feature]

            got, *pose = self.getPoseFromCorrespondence(P, N, F, FN)
            if got:
                return pose
        return None

    
    def getPoseFromCorrespondence(self, P, N, F, FN):
        R = N @ np.linalg.inv(FN)
        R = Rotation.match_vectors(N.T, FN.T)[0].as_dcm()
        u, s, vt = np.linalg.svd(R)
        if np.any(np.abs(s ** 2 - 1) > 0.01):
            #print('Cant rotate')
            return False, None, None
        #TODO:I am pretty sure these two lines are equivalent
        b = np.sum(P * N, axis = 0) - np.sum(F * FN, axis = 0)
        #b = np.array((p1.dot(n1) - (R@f1).dot(n1), p2.dot(n2) - (R@f2).dot(n2), p3.dot(n3) - (R@f3).dot(n3)))
        origin = np.linalg.solve(N.T, b)
        if np.any(np.abs(origin)) > 100:
            return False, None, None


        relative = (P.T - origin.reshape((1, 3))) @ R
        # try:
        #TODO: This is horribly inefficient for just checking 3 points.
        # Can be optimized by looking at whether or not the point lies inside of the
        # face that it is supposed to.
        distance = ProximityQuery(self.trimesh).signed_distance(relative)
        #TODO: This tolerance should be configurable.
        if np.any(np.abs(distance) > 0.002):
            #print('Rejected by distance')
            return False, None, None
    # except Exception e:
        #TODO: Figure out why this fails sometimes.
        # print(relative)
        #return False, None, None
        
        return True, origin, R


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
        return
            

    def _compileFeatures(self, **kwargs):
        N = kwargs.get('N')
        n = len(self.Faces)
        self.Features = []
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    ijk = np.array((i,j,k))
                    ns = self.Normals[ijk]
                    ss = self.Sizes[ijk]

                    if np.linalg.cond(ns) > 1e5:
                        continue
                    self.Features.append((np.sum(ss, keepdims= False), ijk))

        self.Features = sorted(self.Features, reverse = True, key = lambda x : x[0])[:N]
        return
    

    def _cacheFeatures(self, **kwargs):
        name = self.getCacheName(kwargs)
        print(f'Writing to feature cache {name}')
        with open(name, 'bw') as fout:
            pickle.dump(self.Features, fout)
        return


    def getCacheName(self, kwargs):
        name = '-'.join(f'{k}_{v}' for k,v in kwargs.items())
        version = '0.1'
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
