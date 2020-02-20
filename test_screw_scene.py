import numpy as np
import matplotlib.pyplot as plt
import json
from scipy.spatial.transform import Rotation
from mpl_toolkits.mplot3d import Axes3D
from Mesh import Mesh
from ModelFinder import ModelFinder
import point_cloud_utils as pcu
import Verbosifier

if __name__ == '__main__':
    Verbosifier.enableVerbosity()
    with open('Models/SCrewScene.json') as fin:
        cloud = []
        screwCloud = np.array(json.load(fin))
        for p in screwCloud:
            if not np.any(np.isnan(np.array(p))) and np.linalg.norm(np.array(p)-np.array((0,0,0.4)))< 0.3:
                cloud.append(p)
        cloud = np.array(cloud)
        fullCloud = cloud[np.random.choice(range(len(cloud)), len(cloud))]

    mesh = Mesh('Models/ToyScrew-Yellow.stl')
    mesh.compileFeatures(N = 10)
    finder = ModelFinder(mesh)
    mask = ModelFinder.voxelFilter(fullCloud, size = 0.005)
    cloudNormals = pcu.estimate_normals(fullCloud, 5)
    cloud, cloudNormals = fullCloud[mask], cloudNormals[mask]


    fig = plt.figure()
    ax = fig.gca(projection = '3d')

    ax.scatter(cloud[:,0], cloud[:,1], cloud[:,2])
    count = 0
    for o, r in finder.findInCloud(cloud, cloudNormals):
        print(o, r)
        faces = (mesh.Faces @ r.T) + o
        ax.scatter(faces[:,0], faces[:,1], faces[:,2], c = 'r')
        if count > 10:
            break
        count += 1
    plt.show()



