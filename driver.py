#!/usr/bin/python

import settings as S
import dist
from pylab import scatter, clf, show, semilogy, axes, errorbar, figure, savefig, boxplot, setp
import random
import copy
import sys
import types
from numpy import array
from multiprocessing import Pool
from util import get_settings_hash



def simulation_params():
    params = S.SimParams()
    RANDOM_SEED = 100                     # Self explanatory

    params.V = 200
    params.D = 2
    params.MULT_NOISE = False
    params.DIST_THRESHOLD = 8
    params.MAX_EDGE_LEN_IN_METER = 30.0
    params.SINGLE_LC = False
    params.EXACT_STRESS = False
    params.TRILATERATION = False
    params.SAMPLINGS = 4
    params.MIN_PERTURB =  1e-12
    params.SS_SAMPLES = 200
    params.SDP_SAMPLE_MIN = 1
    params.SDP_SAMPLE_MAX = 5
    params.SDP_SAMPLE_ENUMERATE_THRESHOLD = 20
    params.SDP_SAMPLE_MAX_ENUMERATE = 4

    params.K_RING = 2
    params.MIN_LOCAL_NBHD = 20
    params.STRESS_VAL_PERC = 2
    params.RANDOM_STRESS = True
    params.ORTHO_SAMPLES = True

    rnd = random.Random()
    rnd.seed(10)
    seeds = [ rnd.randint(0,65536) for i in xrange(20) ]

    output_params = S.OutputParams()
    output_params.SKIP_PLOT = True

    for params.STRESS_SAMPLE in ['global', 'semilocal']:
        for params.FLOOR_PLAN_FN in [None, "space.xml?10-2"]:    
            for params.NOISE_STD, params.PERTURB in [(1e-3, 20), (1e-4, 80)]:
                for params.MAX_DEGREE in xrange(5, 11):
                    for params.RANDOM_SEED in seeds:
                        yield (copy.deepcopy(params), output_params)

THRESHOLD = 1e-2

def simulate_wrapper(item):
    p = item[0]
    besterror = 1e100
    bestresult = None
    bestparam = None
    for i in xrange(1,3):
        newp = copy.deepcopy(p)
        newp.SDP_SAMPLE_MAX = i * p.SDP_SAMPLE_MAX
        result = dist.simulate(newp, item[1], False)
        if besterror > result.pos_error:
            besterror = result.pos_error
            bestresult = result
            bestparam = newp
        if result.pos_error < THRESHOLD:
            break
    return (bestparam, bestresult) 

def scan_for_error(sims, threshold):
    for param, result in sims:
        if result.pos_error > threshold:
            print "Large error: %s\t:%f" % (get_settings_hash(param), result.pos_error)

def go(num_worker):
    if num_workers > 1:
        pool = Pool(num_worker)
        return pool.map(simulate_wrapper, simulation_params(), 1)
    else:
        return map(simulate_wrapper, simulation_params())


def filter_by_params(sims, **q):
    r = []
    for params, result in sims:
        filtered = 0
        for name, values in q.items():
            if not name in dir(params):
                filtered = 1
                break
            
            v = getattr(params, name)
            if isinstance(v, float):
                filtered = 1
                for val in values:
                    if abs(val - v) < S.EPS:
                        filtered = 0
                        break
                if filtered:
                    break
            else:
                if not v in values:
                    filtered = 1
                    break
        if not filtered:
            r.append((params, result))

    return r

def extract_stats(sims, xname, yname):
    return [(getattr(result, xname), getattr(result, yname)) for params, result in sims]
                
def plot_sims(sims):
    """
    X coordinates is always mean degree
    
    Effects of reduced measurement:
   
    for Type in { iso | aniso}
    for Noise in { Low | High }
    Fig: {global|semilocal}{llc_|}pos_error, {llc_|}dis_error
    Fig: {global|semilocal} v_ratio_llc, v_ratio_ltri
    """

    for method, floor_plan_fn in [('iso', None), ('aniso', "space.xml?10-2")]:
        for noise, noise_std, perturb in [('high', 1e-3, 20), ('low', 1e-4, 80)]:
            s1 = filter_by_params(sims,
                                  STRESS_SAMPLE=['global'],
                                  FLOOR_PLAN_FN=[floor_plan_fn],
                                  NOISE_STD=[noise_std],
                                  PERTURB=[perturb])
            s2 = filter_by_params(sims,
                                  STRESS_SAMPLE=['semilocal'],
                                  FLOOR_PLAN_FN=[floor_plan_fn],
                                  NOISE_STD=[noise_std],
                                  PERTURB=[perturb])
            print len(s1)
            print len(s2)

            # pos_error
            g_pos = extract_stats(s1, 'avg_degree', 'pos_error')
            l_pos = extract_stats(s2, 'avg_degree', 'pos_error')

            #llc_pos_error
            g_llc_pos = extract_stats(s1, 'avg_degree', 'llc_pos_error')
            l_llc_pos = extract_stats(s2, 'avg_degree', 'llc_pos_error')

            #ratio
            g_ratio = extract_stats(s1, 'avg_degree', 'v_ratio_llc')
            l_ratio = extract_stats(s2, 'avg_degree', 'v_ratio_llc')
            tri_ratio = extract_stats(s1, 'avg_degree', 'v_ratio_ltri')


            def xs(p): return array(map(lambda q:q[0], p))
            def ys(p): return array(map(lambda q:q[1], p))

            def preperrors(p):
                h = {}
                for x, y in p:
                    ix = int(round(x))
                    if ix not in h:
                        h[ix] = ([x],[y])
                    else:
                        h[ix][0].append(x)
                        h[ix][1].append(y)

                h = list(h.items())
                h.sort(lambda (ix, xys), (ix2, xys2): cmp(ix, ix2))

                if False:
                    meanx, meany, stdx, stdy, minx, miny, maxx, maxy = [],[],[],[],[],[],[],[]
                    for ix, xys in h:
                        xs = array(xys[0])
                        ys = array(xys[1])
                        meanx.append(xs.mean())
                        meany.append(ys.mean())
                        minx.append(xs.min())
                        miny.append(ys.min())
                        maxx.append(xs.max())
                        maxy.append(ys.max())
                        stdx.append(xs.std())
                        stdy.append(ys.std())
                    return meanx, meany, stdx, stdy, minx, miny, maxx, maxy
                else:
                    print map(lambda (ix, xys): len(xys[0]), h)
                    return (list(map(lambda (ix, xys): [xys[1]], h)),
                            list(map(lambda (ix, xys): ix, h)))

                

            def boxp(p, sym, c, adj):
                x, positions = preperrors(p)
                r = boxplot(x, notch = 1, positions = array(positions) + adj, sym=sym)
                setp(r['boxes'], color = c)
                setp(r['medians'], color = c)
                setp(r['whiskers'], color = c)
                setp(r['caps'], color = c)
                setp(r['fliers'], mfc = None)

            figure()
            semilogy()

            if True:
                boxp(g_pos, sym = 'r+', c='red', adj = 0.0)
                boxp(l_pos, sym = 'bx', c='blue', adj = -0.0)
                
                #scatter(xs(g_pos), ys(g_pos), s = 2, c = 'red', linewidth=(0.0), alpha=1)
                #scatter(xs(l_pos), ys(l_pos), s = 2, c = 'blue', linewidth=(0.0), alpha=1)
      
                #scatter(xs(g_llc_pos), ys(g_llc_pos), s = 2, c = 'blue', linewidth=(0.0), alpha=1)
                #scatter(xs(l_llc_pos), ys(l_llc_pos), s = 2, c = 'yellow', linewidth=(0.0), alpha=1)
                pass
            else:
                for s, c in [(g_pos, 'red'), (l_pos, 'green')]: #, (g_llc_pos, 'blue'), (l_llc_pos, 'yellow')]:
                    e = preperrors(s)
                    errorbar(x = e[0], y = e[1], xerr = e[2], yerr=e[3], c = c)

            savefig("%s-%s-plot.eps" % (method, noise))

            if noise == 'low':
                figure()
                boxp(g_ratio, sym = 'r+', c='red', adj = 0)
                boxp(l_ratio, sym = 'gx', c='green', adj = 0)
                boxp(tri_ratio, sym = 'bd', c='blue', adj = 0)
                savefig("%s-ratioplot.eps" % (method))

    #show()

    
if __name__ == "__main__":
    num_workers = 4
    if len(sys.argv) > 1:
        num_workers = int(sys.argv[1])
    sims = list(go(num_workers))
    scan_for_error(sims, 2e-2)
    plot_sims(sims)
        
