# FAMrec
FAMrec is a medial-axis and ICP based method for recognizing an arbitrary mesh in a scene
represented as a point cloud.

We also have an implementation of Efficient RANSAC and FBF-RANSAC (another one of our methods)
in sub-folders of the repository.

We have created some of our own mesh models, but also use the toolkit set (https://free3d.com/3d-model/toy-tool-kit-982573.html) available under a free personal use license and the Princeton ModelNet40 (https://modelnet.cs.princeton.edu/) for use in academic research only.
All CAD models are under the ownership and copyright of the original authors.

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

`pip3 install open3d`

`pip3 install gimpact`

or `conda install -c conda-forge point_cloud_utils`


