## Overview
We present an implementation of our medial-axis and face-based 3D Pose Recognition Algorithm. This algorithm takes
a library of meshes and fits the poses in a provided single point cloud. Our implementation is meant to be used in an online
method, meaning there is an emphasis on performance.

### Motivation
There are many industries where 3D recognition of objects can be useful. For example, autonomous driving can benefit
from being able to quickly recognize and localize objects in the environment. We desire a reliable
3D Object pose recognition for aviation manufacturing, where robotics can be used to manipulate
recognized objects to complete tasks. These environments are semi-structured, meaning the objects (e.g., bolts, screws, tools)
are often known ahead of time, but the precise locations may not be known, warranting classification and pose recognition.
Notably, as much of the development involves composites and lightweight metals, there is little color
differentiation in the environment.

When originally searching for a method for our application, we were unable to find a reliabe
method that could identify objects in our prototype environment. 


Markdown is a lightweight and easy-to-use syntax for styling your writing. It includes conventions for

```markdown
CODE CODE CODE
```

For more details see [GitHub Flavored Markdown](https://guides.github.com/features/mastering-markdown/).

### Approach

### Implementation
#### Required Packages
We provide a full package python package for our implementation. This was tested using python 3.6.

#### Data Structure
We provide several classes that do stuff!
#### Core Algorithm
##### Medial Axis Matching
##### Modified Iterative-Closest Point
Describe the high-level reqts and the gist of ICP. Add pseudocode.

### Results
Basic recognition - Unit testing results
Recognition under noise
Recognition under occlusion

### Issues/Limitations


### Comparisons
We compare our method with two state of the art open-source algorithms: ObjRecRansac and PointNet++. Details of the implementations and the comparisons follow:
#### ObjRecRANSAC
ObjRecRANSAC is a RANSAC-kernel based method that uses random sampling to identify geometry in the environment. The original algorithm was proposed by Papazov et al in 2011 CITE. The key idea is to identify key sets of points that can be used for recognition as part of a RANSAC algorithm.

Note: The implementation used for the comparison can be found here: https://github.com/tum-mvp/ObjRecRANSAC. In order to build, we found that a very specific set of libraries was needed. This implementation is built against PCL and VTK. In order to get it to work, we choose to manually build VTK 5.10 from source and then to manually build PCL 1.8.0 from source against this VTK. Having other system versions of VTK seems to cause intermittent seg faults as there are many levels in which VTK is included as a library and it is difficult to change the CMake to force a particular version.

Both ObjRecRANSAC and our method take a set of meshes (e.g., stl) and find instances in a provided 3D point cloud. As such, it was relatively straight forward
to run a comparison. We created 25 example scenes to test the algorithms. These include 1-3 objects in various configurations as well as some false-positive meshes. 

#### PointNet++
PointNet++ is a state of the art neural-network based approach for object recognition. The algorithm was proposed in Someone et al. CITE. Using a variety of custom pre-processing layers and tensorflow, this approach is trained to recognize objects and their specific classification.

Note: The implementation used for the comparison can be found here: https://github.com/charlesq34/pointnet2. This implementation requires Tensorflow and NVIDIA CUDA Drivers. We were able to build the package using CUDA 9.0 and TensorFlow something. As a neural-net approach, the system required training. We trained using the model40.

#### Future Work
Greater robustness to noise, occlusions, further testing on situations
Based on error, tell the robot to adjust its view for a better recognition?