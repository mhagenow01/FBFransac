import numpy as np
import itertools
from Verbosifier import verbose
from functools import reduce
import matplotlib.pyplot as plt
import copy

EPSILON = 0.00001
class Interval:
    __slots__ = ['Start', 'End']
    def __init__(self, start, end):
        self.Start = start
        self.End = end

    def rightOf(self, other):
        return np.all((self.End - other.End) > EPSILON)

    def leftOf(self, other):
        return np.all((other.Start - self.Start) > EPSILON)

    def uncross(self):
        ''' Returns a list of intervals, each of which is the 
            the projection of the interval onto the respective axis.
        '''
        return [Interval(self.Start[0:1,i:i+1], self.End[0:1,i:i+1]) for i in range(self.Start.shape[1])]

    def __mul__(self, other):
        return Interval(np.concatenate((self.Start, other.Start), axis = 1), 
                        np.concatenate((self.End, other.End), axis = 1))

    def __sub__(self, other):
        if self <= other:
            return []
        if not self & other:
            return [self]
        if self.Start.shape[1] == 1:
            return self._fastSub(other)
        return self._genericSub(other)
    
    def _fastSub(self, other):
        selfL = self.Start[0,0]
        selfR = self.End[0,0]
        otherL = other.Start[0,0]
        otherR = other.End[0,0]
        ret = []
        if otherL - selfL > EPSILON:
            ret.append(Interval(self.Start, other.Start))
        if selfR - otherR > EPSILON:
            ret.append(Interval(other.End, self.End))

        return ret

    def _genericSub(self, other):
        ret = []
        for i in self ^ other:
            if not i & other:
                ret.append(i)
        return ret

    def __and__(self, other):
        ''' Returns an interval that is the intersection of self and other
        '''
        return Interval(np.max((self.Start, other.Start), axis = 0), np.min((self.End, other.End), axis = 0))

    def __or__(self, other):
        ''' Returns the union of two intervals '''
        return Interval(np.min((self.Start, other.Start), axis = 0), np.max((self.End, other.End), axis = 0))

    def __xor__(self, other):
        ''' Returns a tuple of distinct intervals such that the union of the intervals 
            along with self & other results in the union of self and other.
        '''
        ret = []
        intersection = self & other
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
        for i in range(self.Start.shape[1]):
            if (self.Start[0,i] - other.Start[0,i]) < EPSILON:
                return False
            if (other.End[0,i] - self.End[0,i]) < EPSILON:
                return False
        return True

    def __le__(self, other):
        for i in range(self.Start.shape[1]):
            if (self.Start[0,i] - other.Start[0,i]) < -EPSILON:
                return False
            if (other.End[0,i] - self.End[0,i]) < -EPSILON:
                return False
        return True

    def contains(self, p):
        return np.all((p - self.Start) > EPSILON) and np.all((self.End - p) > EPSILON)
    
    def __bool__(self):
        return bool(np.all((self.End - self.Start) > EPSILON))

    def __repr__(self):
        return f'[{self.Start},{self.End}]'

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

    
class DisjointIntervalTree:
    def __init__(self, interval = None):
        self.Children = []
        self.Interval = interval

    def insertInterval(self, intervalTuple):
        if len(intervalTuple) == 0:
            return

        myInterval, rest = intervalTuple[0], intervalTuple[1:]

        l, r = self.getIntersectionRange(myInterval)
        r = min(r, len(self.Children)- 1)
        if r < l:
            newChild = DisjointIntervalTree(myInterval)
            newChild.insertInterval(rest)
            self.Children.insert(l, newChild)
        else:
            # Look for intersections with the right point of the interval
            if self.Children[r].Interval.contains(myInterval.End[0,0]):
                left, right = self.Children[r].split(myInterval.End, rest, right = False)
                self.Children.pop(r)
                self.Children.insert(r, right)
                self.Children.insert(r, left)
                r += 1
            #assert self.isSorted()

            # Look for intersections with the left point of the interval
            if self.Children[l].Interval.contains(myInterval.Start[0,0]):
                left, right = self.Children[l].split(myInterval.Start, rest, right = True)
                self.Children.pop(l)
                self.Children.insert(l, right)
                self.Children.insert(l, left)
                r += 1
            #assert self.isSorted()

            # Insert any free space.
            remainingInterval = myInterval
            for i in reversed(range(l, r + 1)):
                c = self.Children[i]
                # If c is totally contained in remaining interval
                if c.Interval < remainingInterval:
                    left, right = remainingInterval - c.Interval
                    newChild = DisjointIntervalTree(right)
                    newChild.insertInterval(rest)
                    self.Children.insert(i+1, newChild)
                    remainingInterval = left
                elif not remainingInterval < c.Interval:
                    # If the interval to add is not contained in the child
                    # Then there should be 1 side that is hanging over
                    # Or they are the same interval
                    sub = remainingInterval - c.Interval
                    if len(sub) == 1:
                        sub = sub[0]
                        # If the right is hanging over, just make a new child
                        # this is free space
                        if sub.rightOf(c.Interval):
                            newChild = DisjointIntervalTree(sub)
                            newChild.insertInterval(rest)
                            self.Children.insert(i+1, newChild)
                            self.Children[i].insertInterval(rest)
                            break
                        elif sub.leftOf(c.Interval):
                            # If the left is hanging over, leave it to be comapred against the 
                            # intervals left of self.Children[i]
                            remainingInterval = sub
                    else:
                        # If its not (left and right) and its not left and its not right, 
                        # then it must be the same. So, we don't split the child and 
                        # propogate the interval
                        c.insertInterval(rest)
                        break
            #assert self.isSorted()
        #assert self.isSorted()

    def isSorted(self):
        for i,c in enumerate(self.Children[1:]):
            if self.Children[i].Interval.End[0,0] > c.Interval.Start[0,0]:
                print(self.Children[i].Interval, c.Interval)
                return False
        return True

    def split(self, p, intervals, right = True):
        half = copy.copy(self)
        otherHalf = copy.copy(self)
        otherHalf.Children = []
        for c in half.Children:
            if c.Interval & intervals[0]:
                otherHalf.Children.append(c)
        otherHalf.insertInterval(intervals)

        if right:
            left, right = half, otherHalf
        else:
            left, right = otherHalf, half 
        left.Interval = Interval(left.Interval.Start, p)
        right.Interval = Interval(p, right.Interval.End)
        return left, right
                  
    def getIntersectionRange(self, interval):
        if not self.Children:
            return [0, -1]
        # TODO: Make this a binary search
        # l = 0
        # while l < len(self.Children) and self.Children[l].Interval.End[0,0] <= interval.Start[0,0]:
        #     l += 1
        l = self.getLeftIntersectionPoint(interval.Start[0,0])
        # r = len(self.Children) - 1
        # while r >= l and self.Children[r].Interval.Start[0,0] >= interval.End[0,0]:
        #     r -= 1
        r = self.getRightIntersectionPoint(interval.End[0,0])
        
        return [l, r]

    def getLeftIntersectionPoint(self, p):
        lo = 0
        hi = len(self.Children)
        while lo < hi:
            mid = (lo+hi)//2
            if (self.Children[mid].Interval.End[0,0] - p) < EPSILON: lo = mid+1
            else: hi = mid
        return lo
    
    def getRightIntersectionPoint(self, p):
        lo = 0
        hi = len(self.Children)
        while lo < hi:
            mid = (lo+hi)//2
            if (p - self.Children[mid].Interval.Start[0,0]) > EPSILON: lo = mid+1
            else: hi = mid
        return lo



    def allLeaves(self):
        leaves = []
        for c in self.Children:
            leaves.extend(c.intervalProduct())
        return leaves
    
    def intervalProduct(self):
        if not self.Children:
            return [self.Interval]
        
        product = []
        for c in self.Children:
            product.extend(self.Interval * p for p in c.intervalProduct())
        return product



class SegmentTree:
    def __init__(self, intervals):

        elementaryIntervals = SegmentTree.buildElementaryIntervalsFast(intervals)
        #print(elementaryIntervals)
        #return
        self.buildElementaryTree(elementaryIntervals, intervals.shape[2])
        self.insertRealIntervals(intervals)
        #self.pruneLeaves()
        return

    @verbose()
    def buildElementaryTree(self, elementaryIntervals, dimension):
        self.Root = Node(elementaryIntervals, dimension)

    @verbose()
    def insertRealIntervals(self, intervals):
        for i in intervals:
            self.Root.insertInterval(Interval(i[0:1], i[1:2]))
    
    @verbose()
    def pruneLeaves(self):
        self.Root.prune()


    @staticmethod
    @verbose()
    def buildElementaryIntervalsFast(intervals):
        intervals = [Interval(i[0:1,:], i[1:2,:]) for i in intervals]
        tree = DisjointIntervalTree()
        for i in intervals:
            tree.insertInterval(i.uncross())
        return tree.allLeaves()

    def containsPoint(self, p):
        return self.Root.containsPoint(p)

    def stab(self, p):
        return self.Root.stab(p)

    def renderTo2d(self, ax):
        self.Root.renderTo2d(ax)

if __name__ == '__main__':
    pass

