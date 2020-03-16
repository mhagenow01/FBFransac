import Verbosifier
import json
import numpy as np
from EfficientRANSAC.scripts.EfficientRANSACFinder import EfficientRANSACFinder as ERF
import point_cloud_utils as pcu
import matplotlib.pyplot as plt

def flipNormals(cloudNormals):
    for i,n in enumerate(cloudNormals):
        if n[2] > 0:
            cloudNormals[i] = -n

def main():
    Verbosifier.enableVerbosity()
    with open('EfficientRANSAC/Test_Scenes/Cloud_sphere.json') as fin:
        cloud = []
        screwCloud = np.array(json.load(fin))
        for p in screwCloud:
            if not np.any(np.isnan(np.array(p))):
                # if not np.any(np.isnan(np.array(p))) and np.linalg.norm(np.array(p) - np.array((0, 0, 0.4))) < 0.3:
                cloud.append(p)
        cloud = np.array(cloud)
        fullCloud = cloud  # [np.random.choice(range(len(cloud)), len(cloud))]

    erf = ERF()

    cloudNormals = pcu.estimate_normals(fullCloud, k=10, smoothing_iterations=10)
    mask = ERF.voxelFilter(fullCloud, size=0.005)
    cloud, cloudNormals = fullCloud[mask], cloudNormals[mask]
    flipNormals(cloudNormals)
    erf.findInCloud(cloud, cloudNormals)

    # draw sphere
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    u, v = np.mgrid[0:2 * np.pi:20j, 0:np.pi:10j]
    x = np.cos(u) * np.sin(v)
    y = np.sin(u) * np.sin(v)
    z = np.cos(v)
    ax.plot_wireframe(x, y, z, color="r")

if __name__ == '__main__':
    main()


