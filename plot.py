from pylab import *
from matplotlib.collections import LineCollection
from matplotlib.patches import Circle
from settings import *
from util import *
import os

def plot_info(g, L_opt_p, tri, dim_T, tang_var, stress_var, stress_spec, floor_plan, perturb):

    d, v, p, E  = g.d, g.v, g.p, g.E

    if (d != 2):
        print_info("Non 2d. Skip plotting...")
        return

    def draw_tangent_var():
        semilogy()
        plot(xrange(1, 1+len(tang_var)),tang_var)
        axvline(dim_T)
        gca().set_aspect('auto')

    def draw_stress_var():
        semilogy()
        plot(xrange(len(stress_var)), stress_var)
        axvline(g.e - g.gr.dim_T)
        axhline(median(stress_var)*STRESS_VAL_PERC/100)
        #axvline(len(tang_var)*STRESS_PERC/100)
        gca().set_aspect('auto')


    def draw_stress_eig():
        #Graph the spectral distribution of the stress matrix
        if g.gr.dim_K > g.v/10:
            semilogy()
        else:
            loglog()

        plot(xrange(1, 1 + len(stress_spec)), stress_spec)
        axvline(g.gr.dim_K-1)
        gca().set_aspect('auto')


    def draw_geometry():
        point_size = 20*math.sqrt(30)/math.sqrt(v)
        line_width = point_size * 0.06
        
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
                    c = "purple",
                    marker = 'o',
                    zorder = 103,
                    alpha = 0.75)
            scatter(x = tri.p[0,tri.seed].A.ravel(),
                    y = tri.p[1,tri.seed].A.ravel(),
                    s = point_size * 2,
                    linewidth = (0.0),
                    c = "gold",
                    marker = 'o',
                    zorder = 104,
                    alpha = 0.75)
    
        if floor_plan != None:
            gca().add_collection(LineCollection(floor_plan.as_line_collection(), colors = "blue", alpha=1, linewidth = line_width))

        gca().add_collection(LineCollection([
            p.T[e] for e in E], colors = "lightgrey", alpha=0.75, linewidth=line_width))

        if tri != None:
            gca().add_collection(LineCollection(
                [tri.p.T[e] for e in tri.used_edges],
                colors = "red",
                alpha=1,
                linewidth=line_width))

        gca().add_collection(LineCollection([
            (L_opt_p.T[i].A.ravel(), p.T[i].A.ravel())
            for i in xrange(v)], colors = "red", alpha=0.75, linewidth = line_width))

        axis([-0.01, 1.01, -0.01, 1.01])
        gca().set_aspect('equal')


    def save(prefix):
        savefig("%s.eps" % prefix)
        os.system("epstopdf %s.eps" % prefix)
        print_info("%s.{eps|pdf} generated" % prefix)

    fn = "%s/%s" % (DIR_PLOT, get_settings_hash())

    # draw the summary plot
    figure(figsize=(12,8))
    margin = 0.05
    width = 1.0/3.0
    height = (1.0-margin*2)/2.0

    if tang_var != None:
        clf()
        axes([margin, margin, width-margin*2, height-margin*2])
        title("Tangent Sample Variance")
        draw_tangent_var()
    else:
        clf()
        axes([margin, margin, width-margin*2, height-margin*2])
        title("Sub-Stress PCA Variance")
        draw_stress_var()

    axes([margin, height+margin, width-margin*2, height-margin*2])
    title("Stress Matrix Eigenvalues")
    draw_stress_eig()

    axes([margin+width, margin, width*2-margin*2, height*2-margin*2])
    title("Error")
    draw_geometry()

    save(fn+'-summary')
    os.system("gnome-open %s-summary.pdf" %fn)

    #draw tangent or stress var
    figure(figsize=(6,6))
    clf()
    if tang_var != None:
        draw_tangent_var()
        save(fn+'-tangentvar')
    else:
        draw_stress_var()
        save(fn+'-stressvar')
        

    #draw stress
    figure(figsize=(6,6))
    clf()
    draw_stress_eig()
    save(fn+'-stresseig')

    #draw geometry
    figure(figsize=(6,6))
    clf()
    draw_geometry()
    save(fn+'-geometry')
 
    
    

