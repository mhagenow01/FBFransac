from Mesh import Mesh
import numpy as np
import itertools
import functools
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D

def angleAndDistance(n1, n2, vertices, verts):
    v1, v2 = vertices[verts[0]], vertices[verts[1]]
    r = v2 - v1
    d = np.linalg.norm(r)
    r /= d
    return np.arccos(abs(n1.dot(r))), np.arccos(abs(n2.dot(r))), d

def calcAverageEuclideanDistanceFeatureSpace(featurevector,faces_normals_for_feature):
    number_features = np.shape(featurevector)[0]
    distances = []

    # For a random number of points, keep track of the closest set of indices for comparison
    comparison_samples = np.random.choice(range(0, number_features), size=5)

    print ("Calculating Distances Between Features")
    print ("Number of Features: ",number_features)
    count = 0
    closest_indices = []


    for ii in range(0,number_features):
        if (count%10==0):
            print("Progress %:",float(count)/float(number_features)*100)
        count = count + 1
        closest_distance_temp = np.inf
        closest_j = -1
        for jj in range(ii+1,number_features):
            distances.append(np.linalg.norm(featurevector[ii,:]-featurevector[jj,:]))

            # Keep track of closest distance
            if np.linalg.norm(featurevector[ii,:]-featurevector[jj,:])<closest_distance_temp:
                closest_distance_temp=np.linalg.norm(featurevector[ii,:]-featurevector[jj,:])
                closest_j = jj
        if ii in comparison_samples:
            closest_indices.append([faces_normals_for_feature[ii],faces_normals_for_feature[closest_j]])

    print ("Average Distance: ",np.average(np.array(distances)))
    print("Comparison Face/Normal Combos that are not well differentiated:")
    print(closest_indices)

def main():
    mesh = Mesh('Models/ToyScrew-Yellow.stl')
    nFaces = len(mesh.Faces)

    # Downsample to make faster
    down_samples = 15
    mesh_samples = np.random.choice(range(0,nFaces),size=down_samples)
    nFaces = down_samples

    feature_vector = []
    faces_normals_for_feature = []

    print("Calculating Features")

    for i in range(nFaces):
        for j in range(i + 1, nFaces):
            for k in range(j + 1, nFaces):
                faces = mesh.Faces[[mesh_samples[i], mesh_samples[j], mesh_samples[k]]]
                normals = mesh.Normals[[mesh_samples[i], mesh_samples[j], mesh_samples[k]]]
                current_sample_ids = [mesh_samples[i], mesh_samples[j], mesh_samples[k]]

                # Order the faces based on the angle from (1,0,0)
                n_comparison = np.array([1.0, 0.0, 0.0])
                angles = [np.arccos(np.dot(normals[0],n_comparison)),np.arccos(np.dot(normals[1],n_comparison)),np.arccos(np.dot(normals[2],n_comparison))]
                sorted_faces = np.argsort(angles)
                n1, n2, n3 = normals[sorted_faces[0]],normals[sorted_faces[1]],normals[sorted_faces[2]]
                f1, f2, f3 = faces[sorted_faces[0]], faces[sorted_faces[1]], faces[sorted_faces[2]]

                # Calculate Inner Products as one set of features
                ip1 = np.dot(n1/np.linalg.norm(n1),n2/np.linalg.norm(n2))
                ip2 = np.dot(n1/np.linalg.norm(n1),n3/np.linalg.norm(n3))
                ip3= np.dot(n2 / np.linalg.norm(n2), n3 / np.linalg.norm(n3))

                # Distance between f1 and f2
                v_ind1, v_ind2 = mesh.trimesh.faces[[current_sample_ids[sorted_faces[0]], current_sample_ids[sorted_faces[1]]]]
                distances = np.array(list(map(functools.partial(angleAndDistance, n1, n2, mesh.trimesh.vertices),
                                              itertools.product(v_ind1, v_ind2))))
                minDistance = np.min(distances, axis=0)
                maxDistance = np.max(distances, axis=0)
                meanDistance1 = (minDistance + maxDistance) / 2

                # Distance between f1 and f3
                v_ind1, v_ind2 = mesh.trimesh.faces[
                    [current_sample_ids[sorted_faces[0]], current_sample_ids[sorted_faces[2]]]]
                distances = np.array(list(map(functools.partial(angleAndDistance, n1, n3, mesh.trimesh.vertices),
                                              itertools.product(v_ind1, v_ind2))))
                minDistance = np.min(distances, axis=0)
                maxDistance = np.max(distances, axis=0)
                meanDistance2 = (minDistance + maxDistance) / 2

                # Distance between f2 and f3
                v_ind1, v_ind2 = mesh.trimesh.faces[
                    [current_sample_ids[sorted_faces[1]], current_sample_ids[sorted_faces[2]]]]
                distances = np.array(list(map(functools.partial(angleAndDistance, n2, n3, mesh.trimesh.vertices),
                                              itertools.product(v_ind1, v_ind2))))
                minDistance = np.min(distances, axis=0)
                maxDistance = np.max(distances, axis=0)
                meanDistance3 = (minDistance + maxDistance) / 2

                feature_vector.append([ip1, ip2, ip3, meanDistance1, meanDistance2, meanDistance3])
                faces_normals_for_feature.append([n1,n2,n3,f1,f2,f3])

    feature_vector = np.array(feature_vector)

    # print(feature_vector)

    calcAverageEuclideanDistanceFeatureSpace(feature_vector,faces_normals_for_feature)

    # fig = plt.figure()
    # ax = fig.gca(projection='3d')
    #
    # # ax.set_aspect(1)
    # ax.scatter(points[:, 0], points[:, 1], points[:, 2])
    # fig.canvas.draw()
    # s = ((ax.get_window_extent().height / (sizes[:, 2]) * 72. / fig.dpi) ** 2) / 10000000
    # print(s)
    # ax.clear()
    # ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=np.sqrt(sizes[:, 2]))
    # plt.show()


if __name__ == '__main__':
    main()
    