import json
import time
import numpy as np
from scripts.EfficientRANSACFinder import EfficientRANSACFinder as ERF
import point_cloud_utils as pcu
import matplotlib
# matplotlib.use('Agg') # to make it work with pycharm for troubleshooting
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def flipNormals(cloudNormals):
    for i,n in enumerate(cloudNormals):
        if n[2] > 0:
            cloudNormals[i] = -n

def main():
    with open('Test_Scenes/Cloud_sphere.json') as fin:
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

    ####################
    # Plotting Code    #
    ####################

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.set_xlim3d(-0.5, 0.5)
    ax.set_ylim3d(-0.5, 0.5)
    ax.set_zlim3d(-0.5, 0.5)
    ax.scatter(cloud[:, 0], cloud[:, 1], cloud[:, 2], color='blue', alpha=0.2)

    print("Found ",len(found_spheres), " Spheres!")
    print("Found ",len(found_cylinders), " Cylinders!")

    # Plot all of the spheres
    for ii in range(0,len(found_spheres)):
        r = found_spheres[ii][0]
        c = found_spheres[ii][1]
        u, v = np.mgrid[0:2 * np.pi:20j, 0:np.pi:10j]
        x = r*np.cos(u) * np.sin(v)+c[0]
        y = r*np.sin(u) * np.sin(v)+c[1]
        z = r*np.cos(v)+c[2]
        ax.plot_wireframe(x, y, z, color="r")

    print(found_cylinders)

    # Plot all of the cylinders
    for ii in range(0, len(found_cylinders)):
        # Plot points at center facing up, then rotate pts by a-vector
        r = found_cylinders[ii][0] # radius
        c = found_cylinders[ii][1] # centerpoint
        a = found_cylinders[ii][2] # vector of cylinder direction (aka plane direction)
        min_z = found_cylinders[ii][3]
        max_z = found_cylinders[ii][4]

        x = np.linspace(-r, r, 25)
        z = np.linspace(min_z, max_z, 5)
        xm, zm = np.meshgrid(x, z)
        ym = np.sqrt(r ** 2 - xm ** 2)

        # Rotate based on a-vector
        def skew(x):
            # return the skew symmetric matrix used for rotating
            return np.array([[0, -x[2], x[1]],
                             [x[2], 0, -x[0]],
                             [-x[1], x[0], 0]])

        v = np.cross(np.array([0.0, 0.0, 1.0]),a)
        R = np.eye(3) + skew(v) + skew(v) @ skew(v) * 1/(1+np.dot(np.array([0.0, 0.0, 1.0]),a))

        # Rotate all of the meshgrid points
        rotated_pts = R @ np.stack((xm.flatten(), ym.flatten(), zm.flatten()))
        rotated_pts_neg_y = R @ np.stack((xm.flatten(),-ym.flatten(), zm.flatten()))
        xm=rotated_pts[0, :].reshape((5, 25))
        zm=rotated_pts[2, :].reshape((5, 25))
        ym1=rotated_pts[1, :].reshape((5, 25))
        ym2=rotated_pts_neg_y[1, :].reshape((5, 25))

        # Shift based on center
        xm = xm + c[0]
        ym1 = ym1 + c[1]
        ym2 = ym2 + c[1]
        zm = zm + c[2]

        ax.plot_wireframe(xm, ym1, zm, color="g")
        ax.plot_wireframe(xm, ym2, zm, color="g")


    plt.show()

if __name__ == '__main__':
    main()


