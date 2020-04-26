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
Syntax highlighted code block

# Header 1
## Header 2
### Header 3

- Bulleted
- List

1. Numbered
2. List

**Bold** and _Italic_ and `Code` text

[Link](url) and ![Image](src)
```

For more details see [GitHub Flavored Markdown](https://guides.github.com/features/mastering-markdown/).

### Approach

### Implementation
#### Required Packages
We provide a full package python package for our implementation. This was tested using python 3.6.


### Results

### Issues/Limitations

### Comparisons

Your Pages site will use the layout and styles from the Jekyll theme you have selected in your [repository settings](https://github.com/mhagenow01/FBFransac/settings). The name of this theme is saved in the Jekyll `_config.yml` configuration file.
