import math
from numpy import *
from scipy.linalg.basic import *
from math import sqrt, cos, sin, acos

# //assuming IEEE-754(GLfloat), which i believe has max precision of 7 bits
Epsilon = 1.0e-5

class ArcBall:
    def __init__(self):
        self.origin = array([0.0, 0.0, 1.0], dtype=double)
        
    def map(self, v):
        """ maps a 2d point to a 3d point on the sphere"""
        u = array([v[0], v[1], 0], dtype=double)
        u[0] = max(min(u[0], 1.0), -1.0)
        u[1] = max(min(u[1], 1.0), -1.0)
        n = norm(u)
        if n >= 1:
            u /= n
            n = 1.0
        u[2] = max(0, sqrt(1.0 - n * n))
        return u
        
    def SetOrigin(self, p):
        self.origin = self.map(p)
        
    def GetRotationMatrix(self, p):
        spherePoint = self.map(p)
        if norm(self.origin - spherePoint) < Epsilon:
            axis = [1.0, 0.0, 0.0]
            angle = 0.0
        else:
            axis = cross(self.origin, spherePoint)
            angle = acos(max(min(dot(self.origin, spherePoint), 1.0), -1.0))
        return GetAngleAxisRotation(axis[0], axis[1], axis[2], angle)

def GetAngleAxisRotation(x, y, z, a):
    f = 1 / sqrt(x*x+y*y+z*z)
    x *= f
    y *= f
    z *= f
    c, s = cos(a), sin(a)
    return matrix([[c+(1-c)*x*x, (1-c)*x*y-s*z, (1-c)*x*z+s*y, 0],
                   [(1-c)*x*y+s*z, c+(1-c)*y*y, (1-c)*y*z-s*x, 0],
                   [(1-c)*x*z-s*y, (1-c)*y*z+s*x, c+(1-c)*z*z, 0],
                   [0, 0, 0, 1]], dtype = double)
    
    
