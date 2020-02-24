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
        ''' True if the open intervals have an intersection. False otherwise '''
        return not (np.any(self.Start >= other.End) or np.any(other.Start >= self.End))

    def __or__(self, other):
        ''' Returns the union of two intervals '''
        return Interval(np.min((self.Start, other.Start), axis = 0), np.max((self.End, other.End), axis = 0))

    def __mod__(self, other):
        ''' Returns an interval representing the intersection of both intervals
            Assumes that self & other == True.
        '''
        return Interval(np.max((self.Start, other.Start), axis = 0), np.min((self.End, other.End), axis = 0))

    def __xor__(self, other):
        ''' Returns a tuple of distinct intervals such that the union of the intervals 
            along with self % other results in the union of self and other.
        '''
        ret = []
        intersection = self % other
        union = self | other
        d = self.Start.shape[1]
        for combo in itertools.product((-1,0,1), repeat = d):
            if sum(np.abs(combo)) == 0:
                continue
            start = np.zeros_like(self.Start)
            end = np.zeros_like(self.End)
            for i, v in enumerate(combo):
                if v == -1:
                    start[0,i] = union.Start[0,i]
                    end[0,i] = intersection.Start[0,i]
                if v == 0:
                    start[0,i] = intersection.Start[0,i]
                    end[0,i] = intersection.End[0,i]
                if v == 1:
                    start[0,i] = intersection.End[0,i]
                    end[0,i] = union.End[0,i]
            i = Interval(start, end)
            if self & i or other & i:
                ret.append(i)
                
        return ret

    def __lt__(self, other):
        return np.all(self.Start > other.Start) and np.all(self.End < other.End)

    def __le__(self, other):
        return np.all(self.Start >= other.Start) and np.all(self.End <= other.End)


    def contains(self, p):
        return not (np.any(p < self.Start) or np.any(p > self.End))
    
    def __repr__(self):
        return f'[{self.Start},{self.End}]'
    
    def __mul__(self, other):
        return Interval(np.concatenate((self.Start, other.Start), axis = 1), 
                        np.concatenate((self.End, other.End), axis = 1))

class Node:
    def __init__(self, intervals, dimension = 1, splitDimenion = 0):
        self.Count = 0
        self.Intervals = []
        self.Left = None
        self.Right = None

        if len(intervals) == 1:
            self.Interval = intervals[0]
        else:
            intervals.sort(key = lambda i: i.Start[0, splitDimenion])

            middle = int(np.floor((len(intervals) - 1) / 2))
            forLeft = intervals[:middle+1]
            forRight = intervals[middle+1:]
            if len(forLeft) > 0:
                self.Left = Node(forLeft, dimension, (splitDimenion + 1) % dimension)
                self.Interval = self.Left.Interval
            if len(forRight) > 0:
                self.Right = Node(forRight, dimension, (splitDimenion + 1) % dimension)
                self.Interval = self.Right.Interval

            if self.Left is not None and self.Right is not None:
                self.Interval = self.Left.Interval | self.Right.Interval
        return


    def insertInterval(self, interval):
        self.Count += 1
        if self.Interval <= interval:
            self.Intervals.append(interval)
        else:
            if self.Left is not None and self.Left.Interval & interval:
                self.Left.insertInterval(interval)
            if self.Right is not None and self.Right.Interval & interval:
                self.Right.insertInterval(interval)
        return

    def prune(self):
        if self.Left is not None:
            if self.Left.Count == 0:
                self.Left = None
            else:
                self.Left.prune()
        
        if self.Right is not None:
            if self.Right.Count == 0:
                self.Right = None
            else:
                self.Right.prune()

    def containsPoint(self, p):
        if self.Interval.contains(p):
            if len(self.Intervals) > 0:
                return True
            if self.Left is not None and self.Left.containsPoint(p):
                return True
            if self.Right is not None and self.Right.containsPoint(p):
                return True
        return False

    def hasIntersection(self, interval):
        if self.Interval & interval:
            if len(self.Intervals) > 0:
                return True
            if self.Left is not None and self.Left.hasIntersection(interval):
                return True
            if self.Right is not None and self.Right.hasIntersection(interval):
                return True
        return False

    def stab(self, p):
        intervals = []
        if self.Interval.contains(p):
            if len(self.Intervals) > 0:
                intervals += self.Intervals
            if self.Left is not None:
                intervals += self.Left.stab(p)
            if self.Right is not None:
                intervals += self.Right.stab(p)
        return intervals
    
    def intersectInterval(self, interval):
        intervals = []
        if self.Interval & interval:
            if len(self.Intervals) > 0:
                intervals += self.Intervals
            if self.Left is not None:
                intervals += self.Left.intersectInterval(interval)
            if self.Right is not None:
                intervals += self.Right.intersectInterval(interval)
        return intervals

    def size(self):
        return 1 + sum(c.size() for c in [self.Left, self.Right] if c is not None)

    def renderTo2d(self, ax):
        offset = self.Interval.End - self.Interval.Start
        w = offset[0,0]
        h = offset[0,1]
        color = 'none'
        if self.Intervals:
            color = 'blue'
            ax.add_artist(plt.Rectangle(self.Interval.Start.T, w, h, facecolor=color, edgecolor='black'))
        for c in [self.Left, self.Right]:
            if c is not None:
                c.renderTo2d(ax)

    
class SegmentTree:
    def __init__(self, intervals):

        elementaryIntervals = self.buildElementaryIntervalsFast(intervals)
        self.Root = Node(elementaryIntervals, intervals.shape[2])
        for i in intervals:
            self.Root.insertInterval(Interval(i[0], i[1]))
        self.Root.prune()
        return

    @staticmethod
    def getPositionInList(elementaryIntervals, interval):
        if len(elementaryIntervals) == 0:
            return 0
        low = 0
        high = len(elementaryIntervals)
    
        while high > low + 1:
            mid = (high + low) // 2
            if elementaryIntervals[mid].Start[0,0] < interval.Start[0,0]:
                low = mid
            else:
                high = mid
        if elementaryIntervals[low].Start[0,0] < interval.Start[0,0]:
            low += 1
        return low

    def buildElementaryIntervalsFast(self, intervals):
        intervals = [Interval(i[0:1,:], i[1:2,:]) for i in intervals]
        intervals.sort(key = lambda x: x.Start[0,0])
        elementaryIntervals = [intervals.pop(0)]
        while len(intervals) > 0:
            toAdd = intervals.pop(0)
            i = self.getPositionInList(elementaryIntervals, toAdd)
            i = max(0,i)
            while i < len(elementaryIntervals):
                e = elementaryIntervals[i]
                if e.Start[0,0] > toAdd.End[0,0]:
                    break
                if toAdd & e:
                    elementaryIntervals.remove(e)
                    intersection = toAdd % e
                    j = i
                    while j < len(elementaryIntervals) and elementaryIntervals[j].Start[0,0] > intersection.Start[0,0]:
                        j += 1
                    elementaryIntervals.insert(j, intersection)
                    for other in toAdd ^ e:
                        intervals.append(other)
                    break
                i += 1
            else:
                elementaryIntervals.insert(i,toAdd)
        return elementaryIntervals
                

    def containsPoint(self, p):
        return self.Root.containsPoint(p)

    def stab(self, p):
        return self.Root.stab(p)

    def renderTo2d(self, ax):
        self.Root.renderTo2d(ax)


def imshowTree(tree):
    plt.figure()
    x = np.arange(-4, 4, 0.01)
    y = np.arange(-4, 4, 0.01)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)
    for i, _x in enumerate(x):
        for j, _y in enumerate(y):
            Z[i,j] = 1 if tree.containsPoint(np.array((_x, _y))) else 0
    plt.imshow(Z.T, origin = 1, extent=[-4,4,-4,4])



if __name__ == '__main__':
    d = 10
    n = 100000
    e = 0.001
    error = np.array((e) * d)
    intervals = [
        [n - error, n + error] for n in np.random.randn(n,d)
    ]
    # intervals = [
    #     [[0,0], [1,1]],
    #     [[0.5, 0.5], [2,2]]
    # ]
    # dx = 0.05
    # intervals = [
    #     [[x,x], [x+2*dx, x+2*dx]] for x in np.arange(-2, 2, dx)
    # ]
    intervals = np.array(intervals)
    s = time.time()
    tree = SegmentTree(intervals)
    print(f'Compile time with {n}-{d} dimensional boxes of size {e}: {time.time() - s}')

    fig = plt.figure()
    ax = fig.gca()
    ax.set_xlim([-4,4])
    ax.set_ylim([-4,4])
    s = time.time()
    for q in np.random.randn(10000,d):
        tree.containsPoint(q)
    print(f'Query time 10k points: {time.time() - s}')
    #tree.renderTo2d(ax)
    # imshowTree(tree)
    # x = np.mean(intervals[:,:,0], axis = 1)
    # y = np.mean(intervals[:,:,1], axis = 1)
    #plt.scatter(x, y)
    #plt.show()  
    

