from pylab import *
from matplotlib.collections import LineCollection
from matplotlib.patches import Circle
from util import *
import os
import settings as S
import substress, stress

def plot_geo(hashcode, output_params):
    graph_fn = 'cache/graph-%s.cache' % hashcode
    sdpout_fn = 'cache/graph-%s.txt-sdpout' % hashcode
    geo_prefix = 'cache/graph-%s' % hashcode

    f = open(graph_fn, 'rb')
    g = cPickle.load(f)
    f.close()

    d, v, p, E  = g.d, g.v, g.p, g.E

    opt = matrix(zeros((2,v)))
    
    f = open(sdpout_fn, 'rb')
    anchors = map(int, f.readline().split())
    
    for i in xrange(v):
        fields = f.readline().split()
        opt[0,i] = float(fields[0])
        opt[1,i] = float(fields[1])
    f.close()

    colors = ['black',
              'orangered',
              'limegreen',
              'goldenrod',
              'magenta',
              'chocolate']
    markers = [('s', 0),
               ('o', 0),
               ('d', 0),
               ('p', 0),
               ('^', 0),
               ('>', 0),
               ('v', 0),
               ('<', 0),
               ('+', 1),
               ('x', 1)]
    markers = [('o', 0)]
    point_size = output_params.PLOT_POINT_SIZE*math.sqrt(30)/math.sqrt(v)
    line_width = point_size * 0.5

    def draw_geometry2d(rect, t = None):
        axes(rect)
        if t:
            title(t)

        # draw floor plan
        if g.floor_plan != None:
            gca().add_collection(LineCollection(g.floor_plan.as_line_collection(), colors = "blue", alpha=1, linewidth = line_width/2))

        # draw graph edges. edges connecting two vertices in the same
        # linked component are drawn using a color representing the
        # linked components. The rest of the edges are grey.
        g.lcs = stress.detect_LC_from_kernel(g, g.gr.K_basis)
            
        lcs = [set(lc) for lc in g.lcs]

        gca().add_collection(LineCollection([p.T[e] for e in E],
                                            colors = "lightgrey",
                                            alpha=0.75, linewidth=line_width/2))
        
        scatter(x = p[0,anchors].A.ravel(),
                y = p[1,anchors].A.ravel(),
                s = point_size*15,
                linewidth = markers[0][1]*line_width/2,
                color = 'blue',
                marker = 'd',
                zorder = 102,
                alpha = 1)
                

        for i, lc in enumerate(g.lcs):
            scatter(x = p[0,lc].A.ravel(),
                    y = p[1,lc].A.ravel(),
                    s = point_size,
                    linewidth = markers[i%len(markers)][1]*line_width/2,
                    color = colors[i % len(colors)],
                    marker = markers[i % len(markers)][0],
                    zorder = 101,
                    alpha = 1)

        gca().add_collection(LineCollection([
            (opt.T[i].A.ravel(), p.T[i].A.ravel())
            for i in xrange(v)], colors = "red", alpha=1, linewidth = line_width))

        axis([-0.01, 1.01, -0.01, 1.01])
        gca().set_aspect('equal')

        
    def save(prefix):
        savefig("%s.eps" % prefix)
        print_info("%s.eps generated" % prefix)
        if output_params.CONVERT_PDF:
            print_info("%s.pdf generated" % prefix)
            os.system("epstopdf %s.eps" % prefix)

    #draw geometry

    fs = (output_params.PLOT_SIZE,output_params.PLOT_SIZE)
    dim = 1 - output_params.MARGIN*1.5
    rect = [output_params.MARGIN,output_params.MARGIN,dim,dim]

    figure(figsize=fs)
    draw_geometry2d(rect)
    save(geo_prefix)


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print "Usage: %s graphHash" % sys.argv[0]
    else:
        op = S.OutputParams()
        op.CONVERT_PDF = True
        plot_geo(sys.argv[1], S.OutputParams())
    
    
