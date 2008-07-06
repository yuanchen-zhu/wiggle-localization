from xml.dom.ext import *
from xml.dom.ext.reader import Sax2
from xml import xpath
from numpy import *
from geo import *
from util import *

def read_xys(nodes):
    p = asmatrix(zeros((2, len(nodes)), 'd'))
    for i, q in enumerate(nodes):
        p[0, i] = float(q.getAttribute("x"))
        p[1, i] = float(q.getAttribute("y"))
    return p

def read_extents(nodes):
    p = asmatrix(zeros((2, len(nodes)*2), 'd'))
    for i, q in enumerate(nodes):
        p[0, i*2] = float(q.getAttribute("minx"))
        p[1, i*2] = float(q.getAttribute("miny"))
        p[0, i*2+1] = float(q.getAttribute("maxx"))
        p[1, i*2+1] = float(q.getAttribute("maxy"))
    return p

def inside_bbox(p, bbox):
    if p[0,0] < bbox[0,0] or p[0,0] > bbox[0,1]:
        return False
    if p[1,0] < bbox[1,0] or p[1,0] > bbox[1,1]:
        return False
    return True;

class Floorplan:
        
    def __init__(self, fn):
        print_info("Loading %s..." % fn)
        f = open(fn)
        r = Sax2.Reader()
        doc = r.fromStream(f)
        f.close()

        # load in the floor
        p = read_xys(xpath.Evaluate('floor[1]/contour/point', doc.documentElement))
        mi, ma = p.min(axis=1), p.max(axis=1)
        ma -= mi
        s = 1.0 / max(ma[0], ma[1])
        T =  scale_matrix(array([s,s])) * translate_matrix(-mi)
        self.floor = (T * homogenous_vectors(p))[:2,:]


        # next get all the spaces / portals / centroids
        self.spaces = []
        self.portals = []
        self.blockers = []
        for i in xrange(xpath.Evaluate('count(space)', doc.documentElement)):
            p = read_xys(xpath.Evaluate('space[%d]/contour/point' % (i+1), doc.documentElement))
            p = (T * homogenous_vectors(p))[:2,:]
            self.spaces.append(p)
            n = p.shape[1]
            e = map(lambda node: float(node.getAttribute("index")),
                    xpath.Evaluate('space[%d]/portal/edge' % (i+1), doc.documentElement))
            self.portals.extend(map(lambda i: [p[:,(i)%n].T.A.ravel(), p[:,(i+1)%n].T.A.ravel()], e))
            e = set(e)
            b = []
            for j in xrange(n):
                if not j in e:
                    b.append([p[:,j], p[:,(j+1)%n]])
            self.blockers.append(b)

            self.space_to_use = None

        # get bounding volumns for all spaces

        self.space_centroids = (T * homogenous_vectors(read_xys(
            xpath.Evaluate('space/contour/centroid', doc.documentElement))))[:2,:]

        self.space_bbox = (T * homogenous_vectors(read_extents(
            xpath.Evaluate('space/contour/extent', doc.documentElement))))[:2,:]
        

    def inside(self, p):
        if self.space_to_use != None:
            sp = self.space_to_use
        else:
            sp = xrange(len(self.spaces))
        
        for i in sp:
            if inside_bbox(p, self.space_bbox[:,i*2:i*2+2]) and inside_poly2d(p, self.spaces[i]):
                return True
        return False

    def intersect(self, u, v):
        minx, miny = min(u[0,0], v[0,0]), min(u[1,0],v[1,0])
        maxx, maxy = max(u[0,0], v[0,0]), max(u[1,0],v[1,0])
        for i, b in enumerate(self.blockers):
            bbox = self.space_bbox[:,i*2:i*2+2]
            if maxx < bbox[0,0] or maxy < bbox[1,0] or minx > bbox[0,1] or miny > bbox[1,1]:
                continue
            for e in b:
                if intersect2d(u, v, e[0], e[1]):
                    return True
        return False
                                    
    def floor_as_line_collection(self):
        n = self.floor.shape[1]
        return [[self.floor.T[i].A.ravel(),
                 self.floor.T[(i+1) % n].A.ravel()] for i in xrange(n)]

    def space_as_line_collection(self, i):
        s = self.spaces[i]
        n = s.shape[1]
        return [[s.T[i].A.ravel(),
                 s.T[(i+1) % n].A.ravel()] for i in xrange(n)]

    def portals_as_line_collection(self):
        return self.portals

    def blockers_as_line_collection(self):
        lc = []
        for b in self.blockers:
            lc.extend([[l[0].T.A.ravel(), l[1].T.A.ravel()] for l in b])
        return lc

    def as_line_collection(self):
        return self.blockers_as_line_collection();

if __name__ == '__main__':

    from matplotlib.collections import LineCollection
    from pylab import *

    figure(figsize=(8,8))

    f = Floorplan('space.xml?10-2')
    min = f.floor.min(axis=1)
    max = f.floor.max(axis=1)
    col=['red','green','blue','yellow','purple','brown','orange']
    
    axis([min[0,0], max[0,0], min[1,0], max[1,0]])
    gca().add_collection(
        LineCollection(f.floor_as_line_collection(), color='blue', alpha=1, linewidth=0.6))

    #gca().add_collection(
    #    LineCollection(f.blockers_as_line_collection(), color='red', alpha=1, linewidth=0.6))

    p = f.space_centroids
    for i in xrange(p.shape[1]):
        text(p[0,i], p[1,i], "%d" % i, horizontalalignment='center', verticalalignment='center', fontsize=6)

    for i in xrange(len(f.spaces)):
        gca().add_collection(
            LineCollection(f.space_as_line_collection(i), color=col[i%len(col)], alpha=1, linewidth=0.3))
    
    ## gca().add_collection(
    ##     LineCollection(f.portals_as_line_collection(), color='white', alpha=0.2, linewidth=0.8, zorder=10))

    savefig("floor.eps")
    
