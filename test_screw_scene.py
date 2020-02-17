import numpy as np
import matplotlib.pyplot as plt
import json
from scipy.spatial.transform import Rotation
from mpl_toolkits.mplot3d import Axes3D
from Mesh import Mesh
from ModelFinder import ModelFinder


if __name__ == '__main__':
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


    fig = plt.figure()
    ax = fig.gca(projection = '3d')

    ax.scatter(fullCloud[:,0], fullCloud[:,1], fullCloud[:,2])
    count = 0
    for o, r in finder.findInCloud(fullCloud):
        faces = (mesh.Faces @ r.T) + o
        ax.scatter(faces[:,0], faces[:,1], faces[:,2], c = 'r')
        if count > 10:
            break
        count += 1
    plt.show()



