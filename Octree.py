import numpy as np 
import time

EPSILON = 0.00001
SPLITTING_COEFFS = np.array(((4, 2, 1)), dtype = np.int)

class Node:
    def __init__(self, center, parent = None):
        self.Points = []
        self.Parent = parent
        self.Center = center
        self.Children = None
        self.Siblings = []
        self.Count = 0
        return

    def expand(self, sizes):
        self.Children = np.empty(8, dtype = Node)
        half = sizes / 2
        for i in range(8):
            center = np.copy(self.Center)
            for j in range(3):
                if (i >> (2 - j)) & 1:
                    center[j] += half[j]
                else:
                    center[j] -= half[j]
            self.Children[i] = Node(center, self)
        return

    def getPoints(self):
        if self.Points:
            for p in self.Points:
                yield p
        elif self.Children is not None:
            for c in self.Children:
                for p in c.getPoints():
                    yield p
        return None

    def updateCounts(self):
        if self.Children is not None:
            self.Count = sum(c.updateCounts() for c in self.Children)
        else:
            self.Count = len(self.Points)
        return self.Count

    def __repr__(self):
        return f'{{Center: {self.Center}, Children: {self.Children}}}'

class Octree:
    def __init__(self, points, minLeafSize):
        self.Points = points
        self.Min = np.min(points, axis = 0) - EPSILON
        self.Max = np.max(points, axis = 0) + EPSILON
        self.Center = (self.Min + self.Max) / 2
        self.Size = self.Max - self.Min
        self.Depth = int(np.floor(np.log2(np.max(self.Size)/minLeafSize)))
        self.Sizes = np.array([self.Size / 2**d for d in range(1, self.Depth + 1)])

        self.N = 0
        self.constructTree()
        self.linkChildren(self.Root)
        self.Root.updateCounts()

        return

    def constructTree(self):
        self.Root = Node(self.Center)

        nodes = np.full(len(self.Points), self.Root)
        centers = np.full((len(self.Points),3), self.Root.Center)

        for size in self.Sizes:
            indexes = np.array(np.floor((self.Points - centers) / size) + 1, dtype = np.int)
            indexes = np.array(np.dot(indexes, SPLITTING_COEFFS), dtype = np.int)
            for i in range(len(nodes)):
                if nodes[i].Children is None:
                    self.N += 8
                    nodes[i].expand(size)
                nodes[i] = nodes[i].Children[indexes[i]]
                centers[i] = nodes[i].Center
                
        for i in range(len(nodes)):
            nodes[i].Points.append(i)
        
        return

    def linkChildren(self, parent, sizeIndex = 0):
        if parent.Children is None:
            return
        for child in parent.Children:
            for other in parent.Children:
                if child is other:
                    continue
                child.Siblings.append(other)
            grandparent = parent.Parent
            if grandparent is not None:
                size = self.Sizes[sizeIndex]
                for uncle in grandparent.Children:
                    if uncle is parent or uncle.Children is None:
                        continue
                    for cousin in uncle.Children:
                        if np.all(np.abs(cousin.Center - child.Center) < 1.25 * size):
                            child.Siblings.append(cousin)
            self.linkChildren(child, sizeIndex + 1)
        return


    def __repr__(self):
        return f'{{Min: {self.Min}, Max: {self.Max}, Size: {self.Size}, Depth: {self.Depth}, N: {self.N}}}'

def main():
    points = np.random.rand(1000, 3) * 2 - 1
    s = time.time()
    tree = Octree(points, 0.05)
    print(time.time() - s)

    print(tree)
    print(tree.Root.Count)
    # for n in tree.Root.Children.flatten():
    #     if n.Children is not None:
    #         for c in n.Children.flatten():
    #             print(f'Node: {c}')
    #             for s in c.Siblings:
    #                 print(f'\tSibling{s}')

if __name__ == '__main__':
    main()