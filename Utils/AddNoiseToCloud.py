import json
import sys
import numpy as np

if __name__ == '__main__':
    cloudFile = sys.argv[1]
    std = float(sys.argv[2])
    name = cloudFile.split('.')[0]
    outFile = f'{name}-{std}.json'

    with open(cloudFile) as fin:
        cloud = np.array(json.load(fin))
        cloud += np.random.normal(0, std, cloud.shape)
        with open(outFile,'w') as fout:
            json.dump(cloud.tolist(), fout)