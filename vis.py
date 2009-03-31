#!/usr/bin/python

import wx, numpy, OpenGL.GLU, OpenGL.GL
from wx.glcanvas import GLCanvas
from numpy import *
from scipy.linalg.basic import *
from scipy.linalg.decomp import rq
from OpenGL.GLU import *
from OpenGL.GL import *
from arcball import ArcBall
import pickle
import sys

class Conf:
    pass

def visualize(filename, start, n = 1):
    f = open(filename, "rb")
    for i in range(start):
        tmp = pickle.load(f)
    try:
        for i in range(n):
            conf = pickle.load(f)
            app = wx.PySimpleApp()
            frame = wx.Frame(None, -1, "Graph visualizer", 
                             wx.DefaultPosition, wx.Size(1024, 768))

            vis = VisCanvas(frame, conf)
            frame.Show()
            app.MainLoop()
    except e:
        pass

    f.close()

def countConf(filename):
    f = open(filename, "rb")
    n = 0
    try:
        while True:
            tmp = pickle.load(f)
            n = n + 1
    except EOFError:
        f.close()
        return n

    f.close()
    return n


class VisCanvas(GLCanvas):
    
    def __init__(self, parent, conf):
        GLCanvas.__init__(self, parent, -1)

        pts = conf.p.T
        center = pts.mean(axis = 0)
        bbMax = pts.max(axis=0)
        bbMin = pts.min(axis=0)
        bbDim = bbMax - bbMin
        self.radius = norm(bbDim) * 0.5
        
        # localToModel translates the input coordinates such that
        # their center lies at the origin
        self.localToModel = identity(4, dtype = double)
        self.localToModel[0:3, 3] = -center

        # modelToWorld is a rotation
        self.modelToWorld = identity(4, dtype = double)
        
        # worldToView pushes the camera to [0, 0, radius * 2]
        self.worldToView = identity(4, dtype = double)
        self.worldToView[2, 3] = -self.radius * 2
        
        self.init = False
        self.rotating = False
        self.zooming = False
        self.arcball = ArcBall()        
        self.v = conf.v
        self.d = conf.d
        self.E = conf.E
        self.p = conf.p.T
        
        self.Bind(wx.EVT_PAINT, self.onPaint)
        self.Bind(wx.EVT_SIZE, self.onSize)
        self.Bind(wx.EVT_MOTION, self.onMotion)
        self.Bind(wx.EVT_LEFT_DOWN, self.onLeftMouseDown)
        self.Bind(wx.EVT_LEFT_UP, self.onLeftMouseUp)
        #self.Bind(wx.EVT_RIGHT_DOWN, self.onRightMouseDown)
        #self.Bind(wx.EVT_RIGHT_UP, self.onRightMouseUp)
        
    # Event handlers

    def onSize(self, event):
    	if self.GetContext():
            self.SetCurrent()
            self.resize()
            self.Refresh()
    	event.Skip()
    
    def onPaint(self, event):
    	dc = wx.PaintDC(self)
    	self.SetCurrent()
    	if not self.init:
            self.initGL()
            self.resize()
            self.init = True
    	self.render()
        
    def onLeftMouseDown(self, event):
        self.oldModelToWorld = self.modelToWorld
        self.arcball.SetOrigin(self.normalizeClientCoord(event.GetPosition()))
        self.rotating = True
    
    def onLeftMouseUp(self, event):
        self.rotating = False
    
    def onMotion(self, event):
        if self.rotating:
            p = self.normalizeClientCoord(event.GetPosition())
            self.modelToWorld =  self.arcball.GetRotationMatrix(p) * self.oldModelToWorld
            self.Refresh(eraseBackground = False)
    
    # Actual heavy-lifters
    def initGL(self):
        glClearColor(0, 0, 0, 0)
        glPointSize(4.0)
        pass

    def resize(self):
        self.size = self.GetClientSize()
        glViewport(0, 0, self.size.width, self.size.height)
        
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        aspect = float(self.size.width) / float(self.size.height)
        gluPerspective(50.0, aspect, 0.1, 30.0)
        
    def normalizeClientCoord(self, p):
        x, y = p
        return ((2 * x - self.size.width) / float(self.size.width) * 0.8,
                (self.size.height - 2 * y) / float(self.size.height) * 0.8)
    
    def render(self):
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        glMultTransposeMatrixd(self.worldToView)
        glMultTransposeMatrixd(self.modelToWorld)
        glMultTransposeMatrixd(self.localToModel)
        
    	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glBegin(GL_LINES)     # draw the cameras as wireframe pyramids
        scale = self.radius /20
        glColor3f(1,1,1)
        for (i,j) in self.E:
            glVertex3fv(self.p[i])
            glVertex3fv(self.p[j])
        glEnd()

        glBegin(GL_POINTS)      # draw the points
        glColor3f(1,0,0)
    	for i in range(len(self.p)):
            glVertex3fv(self.p[i])
    	glEnd()


    	self.SwapBuffers()

if __name__ == '__main__':
    if len(sys.argv) > 1:
        visualize(sys.argv[1], 0, 1)
    else:
        visualize("none-rigid", 0, 4)
