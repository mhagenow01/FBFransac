## Overview
We present an implementation of our medial-axis and face-based 3D Pose Recognition Algorithm. This algorithm takes
a library of meshes and fits the poses in a provided single point cloud. Our implementation is meant to be used as
part of an interactive robot authoring program for manufacturing, meaning there is an emphasis on both performance
and reliability. 

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
method that could identify objects in our prototype environment, which inspired us to try to create
a method. More detail is available in our original proposal ADD LINK.

### Approach
Existing approaches to arbitrary mesh recognition often use either a neural-net or RANSAC-based kernel. In our work, we
have developed a method that uses the medial axis of the mesh to determine hypothesis mesh keypoints and a modified version of
Iterative-closest point (ICP) to determine a more precise mesh pose. In our approach, we tried to use geometric features
wherever possible.

In developing this approach, we built multiple other implementations. First, we developed an Efficient RANSAC program that
could identify spheres and cylinders. We then created our first algorithm Face-Based-Features RANSAC (FBF ransac) which was 
a RANSAC-based kernel that used face to point correspondence instead of point to point typical of other RANSAC methods. Both of
these implementations are also in github, but are not discussed further. More detail on these can be found in our Midterm report (LINK).

### Implementation (FAMrec)
#### Required Packages
We provide a full package python package for our implementation. This was tested using python 3.6. One of our goals
was to create as compact and simple of code as possible to promote longevity. Our code is not yet optimized for performance.

Required Packages: numpy,trimesh,point_cloud_utils,pykdtree,pyntcloud

Suggested package install directions:
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
For FAMrec there are a few main classes that build up the method:
* find_models_in_cloud:
* ModelFinder:
* Mesh:
* ModelProfile:


Additionally, the EfficientRANSAC and FBFRansac folders have their own classes and scripts found in separate folders. There is also a utilities folder that
provides several scripts for converting file formats, etc.
#### Core Algorithm
##### Medial Axis Matching
One of the simplest ways to differentiate two objects is their scale. If you can determine the scale of the geometry in the scene, many objects can be quickly ruled out. One way of getting at this idea of localized scale is through the medial axis of the scene. The medial axis is defined as the set of all points that have two or more equidistant scene points. By searching for points on the medial axis of the scene that have the right distance-to-scene value, it is possible to search for locations with an appropriate scale.

![Medial Axis](https://mhagenow01.github.io/FBFransac/images/medial_axis.png "Medial Axis")
<div align="center"> Figure XYZ: Left) Example of Medial Axis adapted from [xyz] Right) Discretized medial axis representation of a toy screw </div>

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

<div align="center"><img src="https://render.githubusercontent.com/render/math?math=w_i=\frac{1}{1%2Be^{\frac{d_i-R}{dr}}}" style="width:150px"></div>

This ensures that points inside of the target radius are always considered heavily, while maintaining a smooth transition from "relevant points" to "irrelevant points" as they get farther away, which allows for consistent convergence.

Now that we have an algorithm for finding locations of similar geometric scale, we need to profile each object in our database to determine what we are looking for and how that relates to the object. This is done by simply sweeping out a range of radii of appropriate scale (as determined by the size of the bounding box) and looking for medial axis points at that scale. When profiling an object, we use a thickness (dr) equal to 2% of the target radius. 

Once we have generated profiles for our objects, we iteratively sample...

TODO


##### Modified Iterative-Closest Point
When determining the pose of 3D objects from noisy data, it is common to use an Iterative-closest point algorithm 
as a means of refining an initial estimate of the object pose. We extend the core algorithm in two key ways to make
it amenable to our object recognition formulation.

As a breif reminder, the premise of ICP is to complete an optimal rotation and translation in order
to align two sets of points. The sets of points are constructed by finding the closest corresponding
points between the sets. The optimal translation and rotation can be solved as a linear system using SVD.

First, we define a custom weighting function cognizant of our face-matching based approach
to recognition. The core ICP algorithm is designed to compute the optimal rotation and
translation to align two sets of points. Often, these will be simliar types of data (e.g., points from a point cloud and points sampled on a mesh).
In our algorithm, we compare the center points of the mesh faces to the points on the point cloud. Naturally, this prescribes some error
for the matching related to the size of the faces. In order to combat this, we implement a weighting function
that punishes large mesh faces in the fitting. Our weighting function also considers distance between the matching points
as is a common choice in weighting functions. Our final weighting functions combines these two ideas and can be computed as:

<div align="center"><img src="https://render.githubusercontent.com/render/math?math=w_i=(1-\frac{d_i}{d_t})(\frac{A_i}{max_j(A_j)})" style="width:150px"></div>

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
#### General results
We chose to evaluate our method by testing it on a new set of meshes. We use the freely available Toy Toolkit from free3D (https://free3d.com/3d-model/toy-tool-kit-982573.html).
This set is comprised of seven meshes: hammer, pliers, saw, screw, screwdriver, spanning wrench, and wrench.

We created 3 random scenes that contain all seven objects with randomized poses. Each example scene is a point cloud consisting
of 100,000 points. We perform a combinatorial (e.g., 7 choose 3) analysis for each of the scenes and report the confusion matrix
results. We added an additional category to the standard confusion matrix  for 'Misfit True Positive' which is when
the correct model is selected, but the pose is incorrect (more common of competing methods). The results are reported in 
Table xyz. ** Note: While the results may seem to be a low percentage, it is on part with existing methods (e.g., ObjRecRANSAC). More detail
and comparison is found below.**

| Object          | True Positive | Misfit True Positive | False Negative | True Negative | False Positive |
|-----------------|---------------|----------------------|----------------|---------------|----------------|
| Hammer          | 24            | 0                    | 19             | 53            | 9              |
| Pliers          | 10            | 0                    | 26             | 45            | 28             |
| Saw             | 13            | 1                    | 24             | 51            | 16             |
| Screw           | 7             | 5                    | 30             | 60            | 0              |
| Screwdriver     | 10            | 1                    | 32             | 58            | 4              |
| Spanning Wrench | 10            | 0                    | 27             | 49            | 19             |
| Wrench          | 4             | 3                    | 19             | 23            | 83             |

We find that our system is able to recognize objects across a variety of shapes and sizes, though we still believe
the algorithm can be improved as far as reliability. In particular, our method had a challenge with the wrench model.

ADD MORE DISCUSSION.

ADD NOTES ON PERFORMANCE!

#### Recognition with Point Cloud Noise
Recognition under noise
![noise testing](https://mhagenow01.github.io/FBFransac/images/noise_testing.png "Noise Testing")
<div align="center"> Figure XYZ: Noise Testing </div>


#### Recognition under Occlusion
The medial-axis part of our algorithm can still identify geometry under large percentages of occlusion, provided that convex 
sections of the mesh are still visible.
To test our ICP algorithm against occlusions, we artificially remove sections of a screw model and use
ICP to recognize the pose. As a reminder, our algorithm is designed to still function seamlessly for less than 20 percent occlusion. In testing, we have found that our algorithm is robust to small amounts of occlusions (<30%).
Below, we show one example of Occlusion testing against a model of a screw. In this figure, blue points are the point cloud points used for fitting and the green points are an outline of the fit mesh. At around 50 percent occlusion, the mesh fit starts to have notable error. At this point, the principal
axis of the remaining points of the screw starts to align with the mesh principal axis. This is expected behavior for an ICP-type algorithm.
At approximately 80 percent, the mesh fit is incorrect. More discussion about occlusions can be found below in the future work.

![occlusion testing](https://mhagenow01.github.io/FBFransac/images/percent_occlusion_ICP.png "Occlusion Testing")
<div align="center"> Figure XYZ: ICP Occlusion testing for the Toy Screw model from the side angle </div>

### Issues/Limitations


### Comparisons
We compare our method with two state of the art open-source algorithms: ObjRecRansac and PointNet++. Details of the implementations and the comparisons follow:

#### Efficient RANSAC

![Efficient RANSAC Comparison](https://mhagenow01.github.io/FBFransac/images/efficientComparison.png "Efficient RANSAC Comparison")
<div align="center"> Figure XYZ: Efficient RANSAC Comparison </div>

#### ObjRecRANSAC
ObjRecRANSAC is a RANSAC-kernel based method that uses random sampling to identify geometry in the environment. The original algorithm was proposed by Papazov et al. [1] in 2011. The key idea is to identify key sets of points that can be used for recognition as part of a RANSAC algorithm.

Note: The implementation used for the comparison can be found here: https://github.com/tum-mvp/ObjRecRANSAC. In order to build, we found that a very specific set of libraries was needed. This implementation is built against PCL and VTK. In order to get it to work, we choose to manually build VTK 5.10 from source and then to manually build PCL 1.8.0 from source against this VTK. Having other system versions of VTK seems to cause intermittent seg faults as there are many levels in which VTK is included as a library and it is difficult to change the CMake to force a particular version.

Both ObjRecRANSAC and our method take a set of meshes (e.g., stl) and find instances in a provided 3D point cloud. As such, it was relatively straight forward
to run a comparison. We created 25 example scenes to test the algorithms. These include 1-3 objects in various configurations as well as some false-positive meshes. 


| Object          | True Positive | Misfit True Positive | False Negative | True Negative | False Positive |
|-----------------|---------------|----------------------|----------------|---------------|----------------|
| Hammer          | 13            | 21                   | 5              | 6             | 61             |
| Pliers          | 38            | 0                    | 0              | 10            | 57             |
| Saw             | 16            | 27                   | 2              | 10            | 50             |
| Screw           | 0             | 0                    | 45             | 60            | 0              |
| Screwdriver     | 4             | 0                    | 28             | 35            | 41             |
| Spanning Wrench | 0             | 0                    | 28             | 31            | 46             |
| Wrench          | 4             | 29                   | 9              | 14            | 49             |


![Example ObjRecRANSAC](https://mhagenow01.github.io/FBFransac/images/objrecransac_example_results.png "Example ObjRecRANSAC")
<div align="center"> Figure XYZ: Example of ObjRecRANSAC Results </div>

![Confusion Results](https://mhagenow01.github.io/FBFransac/images/confusion_results.png "Confusion Results")
<div align="center"> Figure XYZ: Confusion Results </div>



#### PointNet++
PointNet++ is a state of the art neural-network based approach for object recognition. The algorithm was proposed in Qi et al. [2] in 2017. Using a variety of custom pre-processing layers and tensorflow, this approach is trained to recognize objects and their specific classification.

Note: The implementation used for the comparison can be found here: https://github.com/charlesq34/pointnet2. This implementation requires Tensorflow and NVIDIA CUDA Drivers. We were able to build the package using CUDA 9.0 and TensorFlow something. As a neural-net approach, the system required training. We trained using the model40.

PointNet++ is designed as a classifier, meaning for an input point cloud of a single object, it will return a classification from the labels
used during training. Thus, a direct comparison similar to above is not possible. Instead, we focus on a comparison where we 
use FAMrec as a classifier for a representative set of objects from the same classes that the PointNet++ model was built upon.
We train PointNet++ using the ModelNet40 database from Princeton [xyz] (https://modelnet.cs.princeton.edu/). From the test set of ModelNet40, we
extract 30 representative meshes (We were unable to convert 10 of the classes to a format that works with our mesh importing system). For each of these 30 objects,
we load them 5 times with random orientations in PointNet++ and get the classification. We also load them 5 times with random orientations into FAMrec and
get what object is recognized (Note: FAMrec might return no object or possibly multiple - it is not a classifier). This gives us a biased, but reasonable
metric to compare the methods. Table has the classification results:

| Object    | Classification | Object     | Classification | Object     | Classification |
|-----------|----------------|------------|----------------|------------|----------------|
| bed       | 0.0            | door       | 0.0            | radio      | 0.4            |
| bench     | 0.4            | flower pot | 0.0            | range hood | 0.0            |
| bookshelf | 0.0            | glass box  | 0.0            | sink       | 0.0            |
| bottle    | 0.0            | keyboard   | 0.2            | stairs     | 0.4            |
| bowl      | 0.2            | lamp       | 0.4            | stool      | 0.0            |
| car       | 0.6            | laptop     | 0.0            | tent       | 0.2            |
| chair     | 0.2            | mantel     | 0.2            | toilet     | 0.0            |
| cone      | 0.4            | person     | 0.0            | tv stand   | 0.0            |
| cup       | 0.0            | piano      | 0.0            | vase       | 0.6            |
| curtain   | 0.4            | plant      | 1.0            | wardrobe   | 0.0            |



#### Conclusions and Future Work
Greater robustness to noise, occlusions, further testing on situations
Based on error, tell the robot to adjust its view for a better recognition?


#### References
[1] P. Liu, Y. Wang, D. Huang and Z. Zhang, "Recognizing Occluded 3D Faces Using an Efficient ICP Variant," 2012 IEEE International Conference on Multimedia and Expo, Melbourne, VIC, 2012, pp. 350-355.  
[1] Chavdar   Papazov,   Sami   Haddadin,   Sven   Parusel,   Kai   Krieger,   and   Darius   Burschka.Rigid3d   geometry   matching   for   grasping   of   known   objects   in   cluttered   scenes.The InternationalJournal  of  Robotics  Research,   31(4):538–553,   2012.doi:10.1177/0278364911436019.URLhttps://doi.org/10.1177/0278364911436019.  
[2] Charles R. Qi, Li Yi, Hao Su, and Leonidas J. Guibas. Pointnet++: Deep hierarchical feature learning onpoint sets in a metric space.  InProceedings of the 31st International Conference on Neural InformationProcessing Systems, NIPS’17, page 5105–5114, Red Hook, NY, USA, 2017. Curran Associates Inc. ISBN9781510860964
[xyz] Z. Wu, S. Song, A. Khosla, F. Yu, L. Zhang, X. Tang and J. Xiao. 3D ShapeNets: A Deep Representation for Volumetric Shapes. Proceedings of 28th IEEE Conference on Computer Vision and Pattern Recognition (CVPR2015)  
[xyz]  Demirci M.F. Boluk, A.  Object recognition based on critical nodes.Pattern  Anal  Applic, 22:147–163,2019.  doi:  10.1007/s10044-018-00777-w.  
[xyz] Ruwen Schnabel,  Roland Wahl,  and Reinhard Klein.  Efficient ransac for point-cloud shape detection.Comput. Graph. Forum, 26:214–226, 06 2007.  doi:  10.1111/j.1467-8659.2007.01016.x.  
