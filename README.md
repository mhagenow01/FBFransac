## Overview
We present an implementation of our medial-axis and face-based 3D Pose Recognition Algorithm. This algorithm takes
a library of meshes and fits the poses in a provided single point cloud. Our implementation is meant to be used in an online
method, meaning there is an emphasis on performance.

![implementation teaser](https://mhagenow01.github.io/FBFransac/images/solution_teaser.png "Implementation Teaser")

### Motivation
There are many industries where 3D recognition of objects can be useful. For example, autonomous driving can benefit
from being able to quickly recognize and localize objects in the environment. We desire a reliable
3D Object pose recognition for aviation manufacturing, where robotics can be used to manipulate
recognized objects to complete tasks. These environments are semi-structured, meaning the objects (e.g., bolts, screws, tools)
are often known ahead of time, but the precise locations may not be known, warranting classification and pose recognition.
Notably, as much of the development involves composites and lightweight metals, there is little color
differentiation in the environment.

When originally searching for a method for our application, we were unable to find a reliable
method that could identify objects in our prototype environment. 

![robot and camera setup](https://mhagenow01.github.io/FBFransac/images/nasa_uli_setup.JPG "Robot and Camera Setup")

Markdown is a lightweight and easy-to-use syntax for styling your writing. It includes conventions for

```markdown
CODE CODE CODE
```

For more details see [GitHub Flavored Markdown](https://guides.github.com/features/mastering-markdown/).

### Approach
Existing approaches to arbitrary mesh recognition either use a neural-net or RANSAC based kernel. In our work, we
have developed a method that uses the medial axis to determine hypothesis mesh keypoints and a modified version of
Iterative-closest point (ICP) to determine a more precise mesh pose. In our approach, we try our best to avoid
random sampling and to always rely on geometric features for everything.

### Implementation
#### Required Packages
We provide a full package python package for our implementation. This was tested using python 3.6.

TODO: make this right

required packages: numpy,trimesh,point_cloud_utils,pykdtree,pyntcloud

package install directions:

```
pip3 install numpy
pip3 install rtree
sudo apt-get install libspatialindex-dev
pip3 install trimesh
pip3 install git+git://github.com/fwilliams/point-cloud-utils
pip3 install pykdtree
pip3 install pyntcloud
pip3 install progressbar
pip install open3d
or conda install -c conda-forge point_cloud_utils
```





#### Data Structure
We provide several classes that do stuff!
#### Core Algorithm
##### Medial Axis Matching
One of the simplest ways to differentiate two objects is their scale. If you can determine the scale of the geometry in the scene, many objects can be quickly ruled out. One way of getting at this idea of localized scale is through the medial axis of the scene. The medial axis is defined as the set of all points that have two or more equidistant scene points. By searching for points on the medial axis of the scene that have the right distance-to-scene value, it is possible to search for locations with an appropriate scale.

At the core of this algorithm is the ability to find medial axis points that have *approximately* the radius we are looking for. While there are a number of algorithms to find medial axis points from a geometry, they are generally highly sensitive to noise in the cloud. The need to *robustly* identify a *single* point on the medial axis that is a certain distance from the scene motivates the following, Mean Shift inspired, novel algorithm:

```
PROGRAM FindMedialAxisPoint:
  # Description: Attempts once to find a medial axis 
    point in the scene with a given radius. 
  # Inputs: 
    PointCloud  - The set of 3d points representing the scene
    Radius      - The target radius
    dr          - Noise tolerance
  # Outputs: Position or None
  Position <= Random-Point-Near-Cloud
  
  WHILE iterations < MAX_ITERATIONS 
        && PositionDelta > EPSILON:
    neighbors <= Points-within-(Radius + 2dr)-of-Position
    
    IF WeightedAverage(DistanceTo(neighbors)) ~= Radius
      EXIT
    END
    idealPositions <= Points-(1*Radius)-Away-from-neighbors
    Position <= WeightedAverage(idealPositions)
  END
  
  Position <= None
END
```

Here we use a modified Woods-Saxon distribution as a weighting function to compute the weighted average. 

<img src="https://render.githubusercontent.com/render/math?math=w_i=\frac{1}{1%2Be^{\frac{d_i-R}{dr}}}">

This ensures that points inside of the target radius are always considered heavily, while maintaining a smooth transition from "relevant points" to "irrelevant points" as they get farther away.

Now that we have an algorithm for finding locations of similar geometric scale, we need to profile each object in our database to determine what we are looking for and how that relates to the object. This is done by simply sweeping out a range of radii of appropriate scale (as determined by the size of the bounding box) and 

TODO


##### Modified Iterative-Closest Point
When determining the pose of 3D objects from noisy data, it is common to use an Iterative-closest point algorithm 
as a means of refining an initial estimate of the object pose. We extend the core algorithm in two ke ways to make
it amenable to our object recognition formulation.

ICP reminder

First, we define a custom weighting function cognizant of our face-matching based approach
to recognition. The core ICP algorithm is designed to compute the optimal rotation and
translation to align two sets of points. Often, these will be simliar types of data (e.g., points from a point cloud and points sampled on a mesh).
In our algorithm, we compare the center points of the mesh faces to the points on the point cloud. Naturally, this prescribes some error
for the matching related to the size of the faces. In order to combat this, we implement a weighting function
that punishes large mesh faces in the fitting. Our weighting function also considers distance between the matching points
as is a common choice in weighting functions. Our final weighting functions combines these two ideas and can be computed as:

<img src="https://render.githubusercontent.com/render/math?math=w_i=(1-\frac{d_i}{d_t})(\frac{A_i}{max_j(A_j)})">

where <img src="https://render.githubusercontent.com/render/math?math=d_i"> is the distance between the face and the closest point in the point cloud, 
<img src="https://render.githubusercontent.com/render/math?math=d_t"> is the maximum distance considered
in the ICP algorithm, <img src="https://render.githubusercontent.com/render/math?math=A_i"> is the area of the mesh face,
and <img src="https://render.githubusercontent.com/render/math?math=max_j(A_j)"> is the largest face in the mesh.

Second, we implement a random restarts framework around the ICP algorithm. The medial axis reliably gives candidate
positions for the mesh, but not rotation. In order to prevent the ICP from getting caught in local minima during iterations,
we use a random restarts approach where each restart is given a random orientation (drawn from the Haar distribution for SO(3)).

Finally, to provide a level of robustness to Occlusion that is common particularly when point clouds
are constructed from a single image, we implement the occlusion method described in [1] . During each ICP iteration
only a percentage of the closest corresponding face-point combinations are selected. From experimental tuning, we choose to use
80 percent of the faces, which allows for some Occlusion-handling while not skewing results for non-occluded objects.

##### Examples
We provide a main interface that allows for specification of the point cloud and the meshes. It will find
mesh instances in the point cloud and display results using Open3D. Note: The first time you run for a particular mesh,
it will need to run the mesh pre-processing which can take 1-2 minutes.

### Results
Basic recognition - Unit testing results
#### General results and General Performance
Recognition under noise
Recognition under occlusion
#### Recognition under Occlusion
TODO: kevin - does medial axis get effected by occlusions
To test our ICP algorithm against occlusions, we artificially remove sections of a screw model and use
ICP to recognize the pose. In general, we have found that our algorithm is robust to small amounts of occlusions (<30%).
Below, we show one example of Occlusion testing. At around 50 percent occlusion, the mesh fit starts to have notable error.
At approximately 80 percent, the mesh fit is incorrect.

![occlusion testing](https://mhagenow01.github.io/FBFransac/images/percent_occlusion_ICP.png "Occlusion Testing")

### Issues/Limitations


### Comparisons
We compare our method with two state of the art open-source algorithms: ObjRecRansac and PointNet++. Details of the implementations and the comparisons follow:
#### ObjRecRANSAC
ObjRecRANSAC is a RANSAC-kernel based method that uses random sampling to identify geometry in the environment. The original algorithm was proposed by Papazov et al. [1] in 2011. The key idea is to identify key sets of points that can be used for recognition as part of a RANSAC algorithm.

Note: The implementation used for the comparison can be found here: https://github.com/tum-mvp/ObjRecRANSAC. In order to build, we found that a very specific set of libraries was needed. This implementation is built against PCL and VTK. In order to get it to work, we choose to manually build VTK 5.10 from source and then to manually build PCL 1.8.0 from source against this VTK. Having other system versions of VTK seems to cause intermittent seg faults as there are many levels in which VTK is included as a library and it is difficult to change the CMake to force a particular version.

Both ObjRecRANSAC and our method take a set of meshes (e.g., stl) and find instances in a provided 3D point cloud. As such, it was relatively straight forward
to run a comparison. We created 25 example scenes to test the algorithms. These include 1-3 objects in various configurations as well as some false-positive meshes. 

#### PointNet++
PointNet++ is a state of the art neural-network based approach for object recognition. The algorithm was proposed in Qi et al. [2] in 2017. Using a variety of custom pre-processing layers and tensorflow, this approach is trained to recognize objects and their specific classification.

Note: The implementation used for the comparison can be found here: https://github.com/charlesq34/pointnet2. This implementation requires Tensorflow and NVIDIA CUDA Drivers. We were able to build the package using CUDA 9.0 and TensorFlow something. As a neural-net approach, the system required training. We trained using the model40.

#### Future Work
Greater robustness to noise, occlusions, further testing on situations
Based on error, tell the robot to adjust its view for a better recognition?


#### References
[1] P. Liu, Y. Wang, D. Huang and Z. Zhang, "Recognizing Occluded 3D Faces Using an Efficient ICP Variant," 2012 IEEE International Conference on Multimedia and Expo, Melbourne, VIC, 2012, pp. 350-355.  
[1] Chavdar   Papazov,   Sami   Haddadin,   Sven   Parusel,   Kai   Krieger,   and   Darius   Burschka.Rigid3d   geometry   matching   for   grasping   of   known   objects   in   cluttered   scenes.The InternationalJournal  of  Robotics  Research,   31(4):538–553,   2012.doi:10.1177/0278364911436019.URLhttps://doi.org/10.1177/0278364911436019.  
[2] Charles R. Qi, Li Yi, Hao Su, and Leonidas J. Guibas. Pointnet++: Deep hierarchical feature learning onpoint sets in a metric space.  InProceedings of the 31st International Conference on Neural InformationProcessing Systems, NIPS’17, page 5105–5114, Red Hook, NY, USA, 2017. Curran Associates Inc. ISBN9781510860964
