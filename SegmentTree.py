import numpy as np
import matplotlib.pyplot as plt 
from ncls import NCLS
import itertools
import time
from cProfile import Profile
import pstats

class Interval:
    __slots__ = ['Start', 'End']
    def __init__(self, start, end):
        self.Start = start
        self.End = end

    @classmethod
    def fromList(cls, l):
        d = len(l)
        interval = Interval(np.zeros((1,d)), np.zeros((1,d)))
        for i in range(d):
            interval.Start[0,i] = l[i].Start[0,0]
            interval.End[0,i] = l[i].End[0,0]
        return interval
    
    def __and__(self, other):
        return not (np.any(self.Start > other.End) or np.any(other.Start > self.End))
    
    def __lt__(self, other):
        return not (np.any(self.Start < other.Start) or np.any(self.End > other.End))

    def contains(self, p):
        return not (np.any(p < self.Start) or np.any(p > self.End))
    
    def __repr__(self):
        return f'[{self.Start},{self.End}]'
    
    def __mul__(self, other):
        return Interval(np.concatenate((self.Start, other.Start), axis = 1), 
                        np.concatenate((self.End, other.End), axis = 1))

    def split(self, other):
        if not self & other:
            return self, other
        return Interval(min(self.Start, other.Start), max(self.Start, other.Start)), \
                Interval(max(self.Start, other.Start), min(self.End, other.End)), \
                Interval(min(self.End, other.End), max(self.End, other.End))

class Node:
    def __init__(self, intervals):
        self.Count = 0
        self.Intervals = []
        self.Children = []

        if intervals.size == 1:
            self.Interval = intervals.flatten()[0]
        else:
            self.Children = np.empty(2 ** len(intervals.shape), dtype = Node)
            middle = np.array(np.floor((np.array(intervals.shape) - 1) / 2), dtype = np.int)
            for childIndex, combo in enumerate(itertools.combinations_with_replacement((True, False), len(intervals.shape))):
                slices = tuple(slice(None, middle[i]+1, 1) if c else slice(middle[i]+1, None, 1) for i,c in enumerate(combo))
                forChild = intervals[slices]
                if forChild.size > 0:
                    self.Children[childIndex] = Node(forChild)
            
            start = np.min([c.Interval.Start for c in self.Children if c is not None], axis = 0)
            end = np.max([c.Interval.End for c in self.Children if c is not None], axis = 0)
            self.Interval = Interval(start, end)
        return

    def insertInterval(self, interval):
        self.Count += 1
        if self.Interval < interval:
            self.Intervals.append(interval)
        else:
            for c in self.Children:
                if c is not None and c.Interval & interval:
                    c.insertInterval(interval)
        return

    def prune(self):
        for i,c in enumerate(self.Children):
            if c is not None:
                if c.Count == 0:
                    self.Children[i] = None
                else:
                    c.prune()


    def containsPoint(self, p):
        if self.Interval.contains(p):
            if len(self.Intervals) > 0:
                return True
            for c in self.Children:
                if c is not None and c.containsPoint(p):
                    return True
        return False

    def size(self):
        return 1 + sum(c.size() for c in self.Children if c is not None)

    def stab(self, p):
        intervals = []
        if self.Interval.contains(p):
            if len(self.Intervals) > 0:
                intervals = intervals + self.Intervals
            for c in self.Children:
                if c is not None:
                    intervals = intervals + c.stab(p)
        return intervals

    def renderTo2d(self, ax):
        offset = self.Interval.End - self.Interval.Start
        w = offset[0,0]
        h = offset[0,1]
        color = 'none'
        if len(self.Children) == 0:
            color = 'blue'
        ax.add_artist(plt.Rectangle(self.Interval.Start.T, w, h, facecolor=color, edgecolor='black'))
        for c in self.Children:
            if c is not None:
                c.renderTo2d(ax)

    
class SegmentTree:
    def __init__(self, intervals):

        elementaryIntervals = self.buildElementaryIntervals(intervals)
        self.Root = Node(elementaryIntervals)
        for i in intervals:
            self.Root.insertInterval(Interval(i[0], i[1]))
        self.Root.prune()
        return

    def buildElementaryIntervals(self, intervals):

        N = []
        allIntervalsSingleDimension = []
        for d in range(intervals.shape[2]):
            allPoints = np.array(sorted(intervals[:,:,d].flatten()))
            allPoints = allPoints.reshape((len(allPoints), 1, 1))
            singleDimensionIntervals = []
            for i in range(len(allPoints) - 1):
                singleDimensionIntervals.append(Interval(allPoints[i], allPoints[i+1]))
            N.append(len(singleDimensionIntervals))
            allIntervalsSingleDimension.append(singleDimensionIntervals)
        allIntervalsSingleDimension = np.array(allIntervalsSingleDimension, dtype = Interval)
        allIntervals = np.empty(N, dtype = Interval)
        dimensions = np.array(list(range(len(allIntervalsSingleDimension))))


        for i in range(allIntervals.size):
            index = np.unravel_index(i, allIntervals.shape)
            intervalsToCross = [allIntervalsSingleDimension[d, index[d]] for d in dimensions]
            crossed = Interval.fromList(intervalsToCross)
            
            allIntervals[index] = crossed
        return allIntervals

    def containsPoint(self, p):
        return self.Root.containsPoint(p)

    def stab(self, p):
        return self.Root.stab(p)

    def renderTo2d(self, ax):
        self.Root.renderTo2d(ax)


if __name__ == '__main__':
    error = np.array((0.1, 0.1, 0.1))
    intervals = [
        [n - error, n + error] for n in np.random.randn(200,3)
    ]
    intervals = np.array(intervals)
    #pr = Profile()
    #pr.enable()
    tree = SegmentTree(intervals)
    #pr.disable()
    #stats = pstats.Stats(pr).sort_stats('cumtime')
    #stats.print_stats()

    fig = plt.figure()
    ax = fig.gca()
    ax.set_xlim([-4,4])
    ax.set_ylim([-4,4])
    s = time.time()
    for q in np.random.randn(1000,3):
        tree.stab(q)
    print(time.time() - s)
    #tree.renderTo2d(ax)
    print(tree.Root.size())
    #plt.show()  
    

