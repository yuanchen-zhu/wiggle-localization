from pylab import *
from matplotlib.collections import LineCollection
from matplotlib.patches import Circle
from util import *
import os
import settings as S
import substress 

def plot_info(g, L_opt_p, tri, dim_T, tang_var, stress_var, stress_spec, perturb):

    d, v, p, E  = g.d, g.v, g.p, g.E

    if S.sSKIP_PLOT:
        print_info("Skip plotting...")
        return

    def draw_tangent_var(rect, t = None):
        axes(rect)
        if t:
            title(t)
        semilogy()
        plot(xrange(1, 1+len(tang_var)),tang_var)
        axvline(dim_T)
        gca().set_aspect('auto')

    def draw_stress_var(rect, t = None):
        axes(rect)
        if t:
            title(t)
        semilogy()
        plot(xrange(len(stress_var)), stress_var)
        axvline(g.e - g.gr.dim_T)
        axvline(substress.PCA_CUTOFF)
        #axvline(len(tang_var)*S.STRESS_PERC/100)
        gca().set_aspect('auto')


    def draw_stress_eig(rect, t = None):
        axes(rect)
        if t:
            title(t)
        #Graph the spectral distribution of the stress matrix
        if g.gr.dim_K > g.v/10:
            semilogy()
        else:
            loglog()

        plot(xrange(1, 1 + len(stress_spec)), stress_spec)
        axvline(g.gr.dim_K-1)
        gca().set_aspect('auto')

    colors = ['black',
              'orangered'    ,
              'limegreen'         ,
              'goldenrod'     ,
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
    point_size = 4*math.sqrt(30)/math.sqrt(v)
    line_width = point_size * 0.25

    def draw_geometry2d(rect, t = None):
        axes(rect)
        if t:
            title(t)
        gca().add_patch(Circle((0.0, 0.0),
                               radius = perturb,
                               linewidth=line_width,
                               fill=False,
                               alpha = 0.5,
                               facecolor = "lightgrey",
                               edgecolor = "black"))
        
        # graph trilateration result
        if tri != None:
            scatter(x = tri.p[0,tri.localized].A.ravel(),
                    y = tri.p[1,tri.localized].A.ravel(),
                    s = point_size,
                    linewidth = (0.0),
                    c = "red",
                    marker = 'o',
                    zorder = 103,
                    alpha = 0.75)
            scatter(x = tri.p[0,tri.seed].A.ravel(),
                    y = tri.p[1,tri.seed].A.ravel(),
                    s = point_size * 2,
                    linewidth = (0.0),
                    c = "red",
                    marker = 'o',
                    zorder = 104,
                    alpha = 0.75)

        # draw floor plan
        if g.floor_plan != None:
            gca().add_collection(LineCollection(g.floor_plan.as_line_collection(), colors = "blue", alpha=1, linewidth = line_width/2))

        # draw graph edges. edges connecting two vertices in the same
        # linked component are drawn using a color representing the
        # linked components. The rest of the edges are grey.
        lcs = [set(lc) for lc in g.lcs]


        gca().add_collection(LineCollection([p.T[e] for e in E],
                                            colors = "lightgrey",
                                            alpha=0.75, linewidth=line_width/2))

        for i, lc in enumerate(g.lcs):
            scatter(x = p[0,lc].A.ravel(),
                    y = p[1,lc].A.ravel(),
                    s = point_size,
                    linewidth = markers[i%len(markers)][1]*line_width/2,
                    color = colors[i % len(colors)],
                    marker = markers[i % len(markers)][0],
                    zorder = 101,
                    alpha = 1)


        if tri != None:
            gca().add_collection(LineCollection(
                [tri.p.T[e] for e in tri.used_edges],
                colors = "red",
                alpha=1,
                linewidth=line_width/2))

        gca().add_collection(LineCollection([
            (L_opt_p.T[i].A.ravel(), p.T[i].A.ravel())
            for i in xrange(v)], colors = "red", alpha=1, linewidth = line_width))

        axis([-0.01, 1.01, -0.01, 1.01])
        gca().set_aspect('equal')

    def draw_geometry3d(rect, t = None):
        import matplotlib.axes3d as p3
        ax = p3.Axes3D(gcf())
        if t:
            title(t)

        # graph trilateration result
        if tri != None:
            ax.scatter3D(tri.p[0,tri.localized].A.ravel(),
                         tri.p[1,tri.localized].A.ravel(),
                         tri.p[2,tri.localized].A.ravel(),
                         s = point_size,
                         linewidth = (0.0),
                         c = "purple",
                         marker = 'o',
                         zorder = 103,
                         alpha = 0.75)
            ax.scatter3D(tri.p[0,tri.seed].A.ravel(),
                         tri.p[1,tri.seed].A.ravel(),
                         tri.p[2,tri.seed].A.ravel(),
                         s = point_size * 2,
                         linewidth = (0.0),
                         c = "gold",
                         marker = 'o',
                         zorder = 104,
                         alpha = 0.75)

        ax.add_lines([[p.T[e[0]].A.ravel(), p.T[e[1]].A.ravel()] for e in E],
                     colors = "lightgrey",
                     alpha=0.75, linewidth=line_width/2)

        lcs = [set(lc) for lc in g.lcs]
        for i, lc in enumerate(g.lcs):
            ax.scatter3D(p[0,lc].A.ravel(),
                         p[1,lc].A.ravel(),
                         p[2,lc].A.ravel(),
                         s = point_size,
                         linewidth = markers[i%len(markers)][1]*line_width/2,
                         color = colors[i % len(colors)],
                         marker = markers[i % len(markers)][0],
                         zorder = 101,
                         alpha = 1)


        if tri != None:
            ax.add_lines([[tri.p.T[e[0]].A.ravel(),
                           tri.p.T[e[1]].A.ravel()]
                          for e in tri.used_edges],
                         color = "red",
                         alpha=1,
                         linewidth=line_width/2)

        ax.add_lines([(L_opt_p.T[i].A.ravel(), p.T[i].A.ravel())
                      for i in xrange(v)],
                     color = "red",
                     alpha=1,
                     linewidth = line_width)

        ax.set_aspect('equal')

    def draw_geometry(rect, t = None):
        if g.d == 2:
            draw_geometry2d(rect, t)
        elif g.d == 3:
            draw_geometry3d(rect, t)
        else:
            print_info("Framework not in 2d or 3d. Skip geometry plot.")
        
    def save(prefix):
        savefig("%s.eps" % prefix)
        os.system("epstopdf %s.eps" % prefix)
        print_info("%s.{eps|pdf} generated" % prefix)

    fn = "%s/%s" % (S.DIR_PLOT, get_settings_hash())

    # draw the summary plot
    figure(figsize=(12,8))
    margin = 0.05
    width = 1.0/3.0
    height = (1.0-margin*2)/2.0

    clf()

    rect = [margin, margin, width-margin*2, height-margin*2]
    if tang_var != None:
        draw_tangent_var(rect, "Tangent Sample Variance")
    else:
        draw_stress_var(rect, "Sub-Stress PCA Variance")

    draw_stress_eig([margin, height+margin, width-margin*2, height-margin*2],
                    "Stress Matrix Eigenvalues")

    draw_geometry([margin+width, margin, width*2-margin*2, height*2-margin*2],
                  "Error")

    save(fn+'-summary')
    os.system("evince %s-summary.pdf&" %fn)

    #draw tangent or stress var
    fs = (S.sPLOT_SIZE,S.sPLOT_SIZE)
    dim = 1 - S.sMARGIN*1.5
    rect = [S.sMARGIN,S.sMARGIN,dim,dim]

    figure(figsize=fs)
    if tang_var != None:
        draw_tangent_var(rect)
        save(fn+'-tangentvar')
    else:
        draw_stress_var(rect)
        save(fn+'-stressvar')
        

    #draw stress
    figure(figsize=fs)
    draw_stress_eig(rect)
    save(fn+'-stresseig')

    #draw geometry
    figure(figsize=fs)
    draw_geometry(rect)
    save(fn+'-geometry')
