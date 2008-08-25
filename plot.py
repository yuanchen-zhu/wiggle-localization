from pylab import *
from matplotlib.collections import LineCollection
from matplotlib.patches import Circle
from settings import *
from util import *

def plot_info(g, L_opt_p, aff_opt_p, tri,
              dim_T, tang_var, stress_spec, floor_plan, stats):

    d, v, p, E  = g.d, g.v, g.p, g.E

    if (d != 2):
        print_info("Non 2d. Skip plotting...")
        return

    figure(figsize=(12,8))
    
    margin = 0.05
    width = 1.0/3.0
    height = (1.0-margin*2)/2.0

    clf()

    axes([0,0,1,1-margin*2], frameon=False, xticks=[], yticks=[])
    title('Noise:%.04f   Perturb.:%.04f   Measurements:%d'
          % (stats["noise"], stats["perturb"], stats["samples"]))
    figtext(width*2-margin, height*2-margin*1.5,
            'aff. err: %.06f\naff. L-err: %.06f'
            % (stats["af g error"], stats["af l error"]),
            ha = "right", va = "top", color = "green")
    figtext(width*3-margin*2, height*2-margin*1.5,
            'err: %.06f\nL-err: %.06f' %
            (stats["l g error"], stats["l l error"]),
            ha = "right", va = "top", color = "red")
    
    
    #Graph the variance of tangent space samples projected to the orthonormal directions.
    axes([margin, margin, width-margin*2, height-margin*2])
    #loglog()
    semilogy()
    plot(xrange(1, 1+len(tang_var)),tang_var)
    axvline(dim_T)

    axhline(median(tang_var)*STRESS_VAL_PERC/100)
    
    #axvline(len(tang_var)*STRESS_PERC/100)
    
    #axis([1, min(dim_T*16, len(E))+1, 1e-4, 1e1])
    gca().set_aspect('auto')
    title("Tang. Var.")

    #Graph the spectral distribution of the stress matrix
    axes([margin, height+margin, width-margin*2, height-margin*2])
    if g.gr.dim_K > g.v/10:
        semilogy()
    else:
        loglog()

    plot(xrange(1, 1 + len(stress_spec)), stress_spec)
    axvline(g.gr.dim_K)
    #axis([1, 1+len(stress_spec), 1e-2, 1e2 ])
    gca().set_aspect('auto')
    title("Agg. Stress Kern. Spec.")

    #Graph the geometry
    axes([margin+width, margin, width*2-margin*2, height*2-margin*2])
    title("Error")
    point_size = 5*math.sqrt(30)/math.sqrt(v)
    line_width = point_size * 0.03
    
    if PLOT_AFFINE_FIT:
        scatter(x = aff_opt_p[0].A.ravel(),
                y = aff_opt_p[1].A.ravel(),
                s = point_size,
                linewidth = (0.0),
                c = "green",
                marker = 'o',
                zorder=99,
                alpha=0.75)
    ## scatter(x = L_opt_p[0].A.ravel(),
    ##         y = L_opt_p[1].A.ravel(),
    ##         s = point_size,
    ##         linewidth=(0.0),
    ##         c = "r",
    ##         marker = 'o',
    ##         zorder=100,
    ##         alpha=0.75)
    scatter(x = p[0].A.ravel(),
            y = p[1].A.ravel(),
            s = point_size,
            linewidth = (0.0),
            c = "b",
            marker = 'o',
            zorder =102,
            alpha=1)

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
    
    axis([-0.2, 1.2, -0.2, 1.2])
    gca().set_aspect('equal')

    gca().add_patch(Circle((0.5, 1.1),
                           radius = stats["perturb"],
                           linewidth=line_width,
                           fill=False,
                           alpha = 0.5,
                           facecolor = "lightgrey",
                           edgecolor = "black"))

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

    if PLOT_AFFINE_FIT:
        gca().add_collection(LineCollection([
            (aff_opt_p.T[i].A.ravel(), p.T[i].A.ravel())
            for i in xrange(v)], colors = "green", alpha=0.75, linewidth= line_width))

    gca().add_collection(LineCollection([
        (L_opt_p.T[i].A.ravel(), p.T[i].A.ravel())
        for i in xrange(v)], colors = "red", alpha=0.75, linewidth = line_width))

    fn = '%s/plot' % DIR_PLOT
    fn += '-v%d'        % v
    fn += '-dt%1.02f'   % PARAM_DIST_THRESHOLD
    fn += '-mn%02d'     % MAX_DEGREE
    fn += '-n%.04f'     % stats["noise"]
    fn += '-p%.04f'     % stats["perturb"]
    fn += '-s%2.02f'    % stats["sampling"]
    fn += '-ks%02d'     % KERNEL_SAMPLES
    fn += '-os%s'       % str(ORTHO_SAMPLES)[0]
    fn += '-ms%s'       % str(MULT_NOISE)[0]           
    fn += '-el%s'       % str(EXACT_LOCAL_STRESS)[0]   
    fn += '-ss%03d'     % SS_SAMPLES                   
    fn += '-wks%s'      % str(WEIGHT_KERNEL_SAMPLE)[0] 
    fn += '-svp%03d'    % STRESS_VAL_PERC              
    fn += '-%s'         % STRESS_SAMPLE
    fn += '-fl%s'       % FLOOR_PLAN_FN
    fn += '-plkp%s'     % str(PER_LC_KS_PCA)[0]
    fn += '-sdps%d'     % SDP_SAMPLE
    fn += '-dsdp%s'     % str(SDP_USE_DSDP)[0]

    import os
    savefig("%s.eps" % fn)
    os.system("epstopdf %s.eps" %fn)
    os.system("rm %s.eps" % fn)
    print_info("%s.pdf generated" % fn)
    os.system("gnome-open %s.pdf" %fn)
