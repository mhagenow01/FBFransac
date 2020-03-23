# FBFransac
FBF-RANSAC is a RANSAC based method for finding an arbitrary mesh in a scene
represented as a point cloud. Previous methods have looked at using a
RANSAC-based method to identify a mesh pose within a scene. Our method differs as it:
* Pushes computing as much as possible to the mesh side which can be done as a one-time step
* Hierarchically ...
* Face-based features rather than relying on similar sampling between a point cloud
and a sampling of the mesh and normals


### Required Packages

**PyFBFRansac**  - python 3.0+
required packages: numpy,trimesh,point_cloud_utils,pykdtree,pyntcloud

package install directions:

`pip3 install numpy`

`pip3 install rtree`

`sudo apt-get install libspatialindex-dev`

`pip3 install trimesh`

`pip3 install git+git://github.com/fwilliams/point-cloud-utils`

`pip3 install pykdtree`

`pip3 install pyntcloud`

`pip3 install progressbar`

`pip install open3d`

or `conda install -c conda-forge point_cloud_utils`


