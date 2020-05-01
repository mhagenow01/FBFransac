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
from ModelProfile import *
import pickle
import os
import gimpact

DENSITY = 500000 / 7

class ModelFinder:
    def __init__(self):
        self.Models = []
        self.ModelProfiles = []
        self.ModelFiles = []
        self.KeyPointGenerators = []
        self.Scene = None
        self.SceneNormals = None
        self.SceneKd = None
        self.MaxDistanceError = 0.003

    @verbose()
    def set_meshes(self, meshFiles, resolution):
        for f in meshFiles:
            self.Models.append(Mesh(f, resolution))
            self.ModelFiles.append(f)
            #self.Models[-1].cacheMeshDistance()
            self.ModelProfiles.append(ObjectProfile.fromMeshFile(f))
    
    @verbose()
    def set_scene(self, cloud):
        self.Scene = cloud
        self.SceneNormals = pcu.estimate_normals(cloud, 10, 3)
        self.SceneKd = KDTree(cloud)
    

    @verbose()
    def findInstances(self):
        ''' Using the defined meshes, keypoint generators, and scene cloud,
            find all of the potential mesh positions. 

            Returns: (the mesh, the scene keypoint, the corresponding mesh keypoint)
        '''
        instances = []
        #Idk, try a few of these?
        for _ in range(1000):
            for m, profile, file in zip(self.Models, self.ModelProfiles, self.ModelFiles):
                r, meshKeyPoints = profile.sampleRadius()
                ind = np.random.choice(range(len(self.Scene)), replace=True)
                startPoint = self.Scene[ind] + np.random.randn(3) * 0.001
                sceneKeyPoint = SupportSphere(startPoint.reshape((1,3)).copy())

                if sceneKeyPoint.iterate(self.Scene, self.SceneKd, r):
                    # TODO: Allow for potentially multiple keypoints per mesh?
                    meshKeyPoint = meshKeyPoints[np.random.choice(range(len(meshKeyPoints)))]
                    *pose, error = self.determinePose(m, meshKeyPoint.X, sceneKeyPoint.X.reshape((1,3)))
                    print(error)
                    valid, score =  self.validatePose(m, pose, error)
                    if valid:
                        self.addInstance(instances, m, pose, file, score)
        return [i[2:] for i in instances]

    @staticmethod
    def addInstance(instances, mesh, pose, file, score):
        g_mesh = gimpact.TriMesh(mesh.trimesh.vertices @ pose[0].T + pose[1], mesh.trimesh.faces.flatten())

        toRemove = set()
        for i in reversed(range(len(instances))):
            if gimpact.trimesh_trimesh_collision(g_mesh, instances[i][0], True):
                if instances[i][1] > score:
                    return
                else:
                    toRemove.add(i)
        for i in toRemove:
            instances.pop(i)
        instances.append((g_mesh, score, mesh, pose, file))

    def determinePose(self, mesh, meshKp, sceneKp):
        ''' Given a mesh and a correspondence between a point in mesh space and
            a point in scene space, determine the best fit pose of the mesh.
        '''
        R = np.eye(3)
        o = sceneKp
        meshFaces = mesh.Faces - meshKp
        R, o, error = self.ICPrandomRestarts(R, o, meshFaces, mesh.Normals, mesh.Sizes)
        if R is None or o is None:
            return None, None

        return R, o - meshKp @ R.T, error

    def validatePose(self, mesh : Mesh, pose, error):
        ''' Given a mesh, pose, and representation of the scene (self.SceneKd), figure out how
            good the pose is at describing the scene.

            Then return True if its good enough, and False otherwise.
        '''
        R, o = pose
        if R is None or o is None:
            return False, None
        return error < self.MaxDistanceError, -error
        nearbyDistances, nearbyPoints_ind = self.SceneKd.query(o.reshape((1,3)), k = 10000, distance_upper_bound = 2*mesh.Radius)
        nearbyPoints_ind = np.array(nearbyPoints_ind)
        nearbyPoints = self.Scene[nearbyPoints_ind[nearbyPoints_ind < len(self.Scene)]]
        maxPoints = DENSITY * mesh.SurfaceArea
        if len(nearbyPoints) < 0.5 * maxPoints:
            return False, None

        nearbyPoints = (nearbyPoints - o) @ R
        distanceToMesh = mesh.distanceQuery(nearbyPoints)
        outliers = np.sum(distanceToMesh > error)
        inliers = np.sum(np.abs(distanceToMesh) <= error)
        # print(outliers, inliers)
        print(maxPoints, outliers, inliers)
        return inliers / maxPoints > 0.2, inliers / maxPoints
        if outliers > 0:
            return False, None
        if inliers < 60:
            return False, None
        return True


    def ICPrandomRestarts(self,R,o,mesh, meshNormals, meshSizes):
        number_restarts = 10
        best_error = np.inf
        max_faces = 100

        # Downsample the faces if necessary
        if len(mesh)>max_faces:
            indices = np.random.choice(len(mesh),max_faces,replace=False)
            mesh=mesh[indices,:]
            meshNormals=meshNormals[indices,:]
            meshSizes=meshSizes[indices,:]

        for ii in range(0,number_restarts):
            R = special_ortho_group.rvs(3) # random restart for R_initial

            o_pert = 0.00*np.array([random.random()-0.5, random.random()-0.5,random.random()-0.5]).reshape((1,3))
            # print(o)
            # print(o_pert)
            # print(o+o_pert)

            R_temp, o_temp, error = self.runICP(R,o+o_pert,mesh,meshNormals, meshSizes)
            print("ICP ",ii," done")

            if error<best_error:
                best_error = error
                best_R, best_o = R_temp, o_temp

        return best_R, best_o, best_error

    def runICP(self, R, o, mesh, meshNormals, meshSizes):
        ''' Given a current pose (R,  o) for a mesh, use ICP to iterate
        and find a better pose that aligns the closest points
        '''
        # Parameters for ICP
        max_iterations = 30 # max iterations for a mesh to preserve performance
        keep_per = 0.8 # percentage to keep for occlusion-handling
        tolerance = 0.001 # when to stop ICP -> cumulative error
        distance_threshold = 0.1 # 10 cm away for closest point TODO: make this based on mesh radius?

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
                return np.eye(3), np.zeros((1,3)), np.inf


            #########################################
            # Ability to add weights to the points  #
            #########################################

            # All ones is no-weighting
            # weights = np.ones((len(s_vals),))

            # # weights based on distance
            weights = (1.0 - (np.abs(distances[closeEnough])/distance_threshold))
            weights_p1 = (1.0 - (np.abs(distances[closeEnough])/distance_threshold)).reshape((len(s_vals),1))
            weights_p2 = (1 - (meshSizes[closeEnough] / np.max(meshSizes))).reshape((len(s_vals), 1))
            weights = np.multiply(weights_p1,weights_p2).reshape((len(s_vals),))
            # # weights based on the size of the mesh faces
            # weights = (meshSizes[closeEnough]/np.max(meshSizes)).reshape((len(s_vals),))

            # # weights based on normals
            # weights = np.abs(np.sum(self.SceneNormals[closest_indices][closeEnough]*(meshNormals[closeEnough] @ R.T),axis=1))


            ########################################
            # Robustness to Occlusions             #
            ########################################

            # Throw out a percentage of outliers
            weight_ind = np.argsort(weights)
            trunc = keep_per*len(s_vals) # percentage of points to keep for occlusion
            s_vals = s_vals[weight_ind[-int(trunc):],:]
            m_vals = m_vals[weight_ind[-int(trunc):],:]
            weights = weights[weight_ind[-int(trunc):]]


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
            # print(" T: ",t)
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
            ind_trunc = np.argsort(distances_temp)[0:int(keep_per*len(distances_temp))]
            closest_points_temp = self.Scene[closest_indices_temp][ind_trunc]
            error = np.max(np.linalg.norm(closest_points_temp-face_points_temp[ind_trunc],axis=1))
            # print("ERROR ", error)
            number_iterations += 1

        # print(number_iterations)
        return R, o, error
