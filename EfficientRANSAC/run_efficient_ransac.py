import json
import time
import numpy as np
from scripts.EfficientRANSACFinder import EfficientRANSACFinder as ERF
import point_cloud_utils as pcu
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def flipNormals(cloudNormals):
    for i,n in enumerate(cloudNormals):
        if n[2] > 0:
            cloudNormals[i] = -n

def main():
    with open('Test_Scenes/Cloud_primitive_playground.json') as fin:
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
    found_spheres, found_cylinders = erf.findInCloud(cloud, cloudNormals)

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.set_xlim3d(-0.5, 0.5)
    ax.set_ylim3d(-0.5, 0.5)
    ax.set_zlim3d(-0.5, 0.5)
    ax.scatter(cloud[:, 0], cloud[:, 1], cloud[:, 2], color='blue', alpha=0.2)
    print("Found ",len(found_spheres), " Spheres!")
    print("Found ",len(found_cylinders), " Cylinders!")

    print("SPHERES!!!!!!!!!!!!!")
    print(found_spheres)
    print("Cylinders!!!!!!!!!!!!!")
    print(found_cylinders)

    # Plot all of the spheres
    for ii in range(0,len(found_spheres)):
        r = found_spheres[ii][0]
        c = found_spheres[ii][1]
        u, v = np.mgrid[0:2 * np.pi:20j, 0:np.pi:10j]
        x = r*np.cos(u) * np.sin(v)+c[0]
        y = r*np.sin(u) * np.sin(v)+c[1]
        z = r*np.cos(v)+c[2]
        ax.plot_wireframe(x, y, z, color="r")

    # Plot all of the cylinders
    for ii in range(0, len(found_cylinders)):
        r = found_cylinders[ii][0]
        c = found_cylinders[ii][1]
        a = found_cylinders[ii][1]
        min_z = found_cylinders[ii][1]
        max_z = found_cylinders[ii][1]

        u, v = np.mgrid[0:2 * np.pi:20j, 0:np.pi:10j]
        x = r * np.cos(u) * np.sin(v) + c[0]
        y = r * np.sin(u) * np.sin(v) + c[1]
        z = r * np.cos(v) + c[2]
        # ax.plot_wireframe(x, y, z, color="r")


    # Plot all of the cylinders
    plt.show()

if __name__ == '__main__':
    main()


