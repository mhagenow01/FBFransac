import numpy as np
import itertools
from Verbosifier import verbose
import Verbosifier
from queue import Queue
from scipy.ndimage import correlate
from collections import defaultdict
from scipy.sparse import lil_matrix
from scipy.sparse.csgraph import floyd_warshall

class Node:
    def __init__(self, index, value):
        self.Adjacent = set()
        self.t_ind = tuple(index)
        self.Index = index
        self.Value = value
        self.Label = None
        self.MatrixIndex = None
    
    def removeFromAdjacent(self):
        for a in self.Adjacent:
            a.Adjacent.remove(self)

    def __hash__(self):
        return hash(self.t_ind)

    def __eq__(self, other):
        return self.t_ind == other.t_ind

    def __repr__(self):
        return repr(self.t_ind)

class Graph:
    def __init__(self):
        self.Nodes = {}

    @staticmethod
    def neighborIndex(shape, ind, offset):
        for combo in itertools.product(*(((-1, 0, 1),) * len(shape))):
            offset = np.array(combo)
            if np.all(offset == 0):
                continue
            neighbor = ind + offset
            if np.all(neighbor >= 0) and np.all(neighbor < shape):
                yield neighbor
        return None

    @verbose()
    def labelConnectedComponents(self):
        for n in self.Nodes.values():
            n.Label = None
        label = 0
        for v in self.Nodes.values():
            if v.Label is None:
                self._labelComponent(v, label)
                label += 1
        return label

    def _labelComponent(self, node, label):
        node.Label = label
        for n in node.Adjacent:
            if n.Label is None:
                self._labelComponent(n, label)
        return
    
    @verbose()
    def connectedComponentCenters(self):
        for n in self.Nodes.values():
            n.MatrixIndex = None

        maxLabel = self.labelConnectedComponents()
        centers = {}
        for l in range(maxLabel):
            nodeList = []
            for n in self.Nodes.values():
                if n.MatrixIndex is None and n.Label == l:
                    n.MatrixIndex = len(nodeList)
                    nodeList.append(n)
            graph = lil_matrix((len(nodeList), len(nodeList)))
            for i, n in enumerate(nodeList):
                for a in n.Adjacent:
                    graph[i, a.MatrixIndex] = 1
            dist_matrix = floyd_warshall(graph, False, False, True)
            centers[i] = nodeList[np.argmin(np.max(dist_matrix, 1))]
        return centers.values()
    
    def maxDistance(self, node):
        seen = set()
        q = Queue()
        maxD = 0
        q.put((node, 0))
        while not q.empty():
            n, d = q.get()
            maxD = max(d, maxD)
            if n in seen:
                continue
            seen.add(n)
            for a in n.Adjacent:
                if a not in seen:
                    q.put((a, d+1))
        return maxD


    def toMatrix(self, m):
        for k,v in self.Nodes.items():
            m[k] = v.Value
        return m

    def toCloud(self, nodes):
        return np.array([n.Index for n in nodes]), np.array([n.Value for n in nodes])

    @verbose()
    def fromMatrix(self, m):
        self.Nodes = {}
        indices = np.array(np.where(m)).T
        shape = np.array(m.shape, dtype = np.int)
        offsets = [np.array(combo) for combo in itertools.product(*(((-1, 0, 1),) * len(shape)))]
        offsets = list(filter(lambda x: not np.all(x == 0), offsets))
        for i in indices:
            t_i = tuple(i)
            if t_i not in self.Nodes.keys():
                node = Node(i, m[t_i])
                self.Nodes[t_i] = node
                for offset in offsets:
                    n = i + offset
                    if np.all(n >= 0) and np.all(n < shape):
                        t_n = tuple(n)
                        if m[t_n] and t_n in self.Nodes.keys():
                            neighbor = self.Nodes[t_n]
                            node.Adjacent.add(neighbor)
                            neighbor.Adjacent.add(node)
        return

    @verbose()
    def prune(self, n, bottom = 20):
        toPrune = set()
        while len(self.Nodes) - len(toPrune) > bottom:
            for k in toPrune:
                self.Nodes[k].removeFromAdjacent()
                self.Nodes.pop(k)
            toPrune = [k for k,v in self.Nodes.items() if len(v.Adjacent) < n-1]
            if len(toPrune) == 0:
                n += 1
        return

    def __iter__(self):
        for v in self.Nodes.values():
            yield v
        return None

    def __repr__(self):
        return '\n'.join(f'{n}: {n.Adjacent}' for n in self.Nodes.values())

def matrixIterate(skeleton, n = 1, minNodes = 5):
    f = np.ones((3,3))
    count = correlate(skeleton.astype(np.int), f, mode='constant')
    c = np.sum(skeleton)
    lastC = None
    while np.sum(skeleton[count >= n]) > minNodes:
        skeleton[count < n] = 0
        count = correlate(skeleton.astype(np.int), f, mode='constant')
        c = np.sum(skeleton)
        print(n,c)
        if c == lastC:
            n+=1
        lastC = c
    return skeleton

if __name__ == '__main__':
    Verbosifier.enableVerbosity()
    g = Graph()
    skeleton = np.ones((20,20))
    g.fromMatrix(skeleton)

    f = np.ones((3,3))

    g.prune(1,5)
    skeleton = matrixIterate(skeleton)
    print(g.toMatrix(np.zeros_like(skeleton)))
    print(correlate(skeleton, f, mode='constant') * skeleton)
    print(skeleton)

    print((g.toMatrix(np.zeros_like(skeleton)) == correlate(skeleton, f, mode='constant') * skeleton).astype(np.int))
    
        