from ModelProfile import *
import matplotlib.pyplot as plt
from matplotlib.animation import ArtistAnimation
import matplotlib.animation as animation
import numpy as np
from pykdtree.kdtree import KDTree
from mpl_toolkits.mplot3d import Axes3D
from Mesh import Mesh

def drawMesh(ax, mesh : Mesh):
    faceOutlines = []
    for face in mesh.trimesh.faces:
        vertices = mesh.trimesh.vertices[face]
        faceOutlines.append(ax.plot(vertices[:,0], vertices[:,1], vertices[:,2], c = 'blue')[0])
    return faceOutlines


def drawSphere(ax, pos, r):
    u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
    x = r * np.cos(u)*np.sin(v) + pos[0,0]
    y = r * np.sin(u)*np.sin(v) + pos[0,1]
    z = r * np.cos(v) + pos[0,2]
    return ax.plot_wireframe(x, y, z, color="r")


if __name__ == '__main__':

    with open('Models\Cloud_ToyScrew-Yellow.json') as fin:
        scene = np.array(json.load(fin))
    kd = KDTree(scene)
    r = 0.0064
    while True:
        ind = np.random.choice(range(len(scene)))
        startPoint = scene[ind] + np.random.randn(1,3) * 0.001
        positions = []

        sphere = SupportSphere(startPoint)
        found = None
        while found is None:
            positions.append(sphere.X.copy())
            found = sphere.update(scene, kd, r)
        positions.append(sphere.X.copy())
        if found:
            break
    artists = []
    fig = plt.figure(figsize = (3,6))
    ax = fig.gca(projection = '3d')
    cloud = ax.scatter(scene[:,0], scene[:,1], scene[:,2], s = 2)
    for p in positions:
        artists.append([
            cloud,
            drawSphere(ax, p, r)
        ])
    ax.set_axis_off()
    ax.view_init(10, -30)
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)
    ani = ArtistAnimation(fig, artists)
    fig = plt.figure(figsize = (3,6))
    ax = fig.gca(projection = '3d')
    faceOutlines = drawMesh(ax, Mesh('Models\ToyScrew-Yellow.stl', 0))
    drawSphere(ax, positions[-1], r)
    ax.set_axis_off()
    ax.view_init(10, -30)
    plt.show()
    ani.save('keypoint_animation.mp4', writer = writer)
    


