import string
from numpy import *
from scipy.linalg.basic import *
from geo import *

class FloorPlan:
    def __init__(self, fn):
        """
        Read 2D floor plan from a file. The format of the file is as
        follows:
        
        n                               # num of polygons
        poly[0].nv
        poly[0].vtx[0..poly[0].nv-1].x
        poly[0].vtx[0..poly[0].nv-1].y
        poly[1].nv
        poly[1].vtx[0..poly[1].nv-1].x
        poly[1].vtx[0..poly[1].nv-1].y
        ...
        poly[n-1].nv
        poly[n-1].vtx[0..poly[n-1].nv-1].x
        poly[n-1].vtx[0..poly[n-1].nv-1].y
       
        """
        f = open(fn, "rb")

        self.np = int(string.split(f.readline())[0])
        self.polys = []
        for i in xrange(self.np):
            nv = int(string.split(f.readline())[0])
            p = asmatrix(zeros((2, nv), 'd'))
            for c in xrange(2):
                p[c,:] = map(float, string.split(f.readline()))
            self.polys.append(p)
        f.close()

    def inside(self, p):
        for poly in self.polys:
            if inside_poly2d(p, poly):
                return False
        return True

    def intersect(self, u, v):
        for poly in self.polys:
            nv = poly.shape[1]
            for j in xrange(nv):
                if intersect2d(u, v, poly[:,j], poly[:,(j+1)%nv]):
                    return True
        return False

    def as_line_collection(self):
        e = []
        for p in self.polys:
            n = p.shape[1]
            lines = [[p.T[i].A.ravel(),
                     p.T[(i+1) % n].A.ravel()] for i in xrange(n)]
            e.extend(lines)
        return e
            
        

        

            
            
        
