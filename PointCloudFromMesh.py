from pyntcloud.io import read_ply
import numpy as np 
import sys
import pyntcloud.io
import pandas as pd
import json



def triangle_area(v1, v2, v3):
    # Cross product returns area of parallelogram for two vectors -> divide by 2 for triangle area
    return 0.5 * np.linalg.norm(np.cross(v2 - v1, v3 - v1), axis = 1)


def pointCloudFromMesh(file, n):
    scene = read_ply(file)
    scene_xyz = scene['points'][['x','y','z']].values
    
    v1_xyz = scene_xyz[scene['mesh']['v1']]
    v2_xyz = scene_xyz[scene['mesh']['v2']]
    v3_xyz = scene_xyz[scene['mesh']['v3']]


    areas = triangle_area(v1_xyz, v2_xyz, v3_xyz)
    prob = areas / areas.sum()

    rand_ix = np.random.choice(range(areas.shape[0]), size = n, p = prob)

    v1_xyz = v1_xyz[rand_ix]
    v2_xyz = v2_xyz[rand_ix]
    v3_xyz = v3_xyz[rand_ix]
    

    u, v = np.random.rand(n, 1), np.random.rand(n, 1)
    needs_flip = u + v > 1
    u[needs_flip] = 1 - u[needs_flip]
    v[needs_flip] = 1 - v[needs_flip]
    w = 1 - (u + v)
    return np.array(v1_xyz * u + v2_xyz * v + v3_xyz * w)


def main(name,n):
    cloud = pointCloudFromMesh(name, int(n))
    cloudFrame = pd.DataFrame({'x': cloud[:, 0], 'y': cloud[:, 1], 'z': cloud[:, 2]})

    pointCloud = pyntcloud.PyntCloud(cloudFrame)

    file_split=name.rpartition('/')

    # Check if file has slashes aka global path or subfolder
    if file_split[1]=='/':
        output_file = file_split[0]+file_split[1]+'Cloud_'+file_split[2]
    else: # otherwise it is just the filename
        output_file = 'Cloud_' + name


    # Save as both a ply and a JSON
    pointCloud.to_file(output_file)
    output_file = output_file.split('.')[0] + '.json'
    lst = cloud.tolist()
    with open(output_file, 'w') as outfile:
        json.dump(lst, outfile)


if __name__ == '__main__':
    # Name of PLY file to convert and number of random samples to take from mesh (n)
    name = sys.argv[1]
    n = sys.argv[2]
    main(name,n)

    