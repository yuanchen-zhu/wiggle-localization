#!/usr/bin/python

import settings as S
from matplotlib import rcParams
rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['Times New Roman']
from matplotlib import rc


import dist
import pylab as P
import random
import copy
import sys
import types
from numpy import array, zeros, linspace, arange,median
from multiprocessing import Pool
from util import get_settings_hash
import math

DEF_PARAMS = S.SimParams()
DEF_PARAMS.V = 200
DEF_PARAMS.D = 2
DEF_PARAMS.MULT_NOISE = False
DEF_PARAMS.DIST_THRESHOLD = 8
DEF_PARAMS.MAX_EDGE_LEN_IN_METER = 30.0
DEF_PARAMS.SINGLE_LC = False

if S.OutputParams.GENERATE_TXT_GRAPH:
    DEF_PARAMS.SINGLE_LC = True

DEF_PARAMS.EXACT_STRESS = False
DEF_PARAMS.TRILATERATION = False
DEF_PARAMS.SAMPLINGS = 4
DEF_PARAMS.MIN_PERTURB =  1e-12
DEF_PARAMS.SS_SAMPLES = 200
DEF_PARAMS.SDP_SAMPLE_MIN = 1
DEF_PARAMS.SDP_SAMPLE_MAX = 5
DEF_PARAMS.SDP_SAMPLE_ENUMERATE_THRESHOLD = 20
DEF_PARAMS.SDP_SAMPLE_MAX_ENUMERATE = 4


DEF_PARAMS.K_RING = 2
DEF_PARAMS.MIN_LOCAL_NBHD = 20
DEF_PARAMS.STRESS_VAL_PERC = 2
DEF_PARAMS.RANDOM_STRESS = True
DEF_PARAMS.ORTHO_SAMPLES = True

#default
DEGREE_NUM_SAMPLES_PER_PARAM = 20
SIZE_NUM_SAMPLES_PER_PARAM = 0
#DEGREE_NUM_SAMPLES_PER_PARAM = 4
#SIZE_NUM_SAMPLES_PER_PARAM = 1

globalCounter = 0

def getCounter():
    global globalCounter
    globalCounter = globalCounter + 1
    return globalCounter

def simulation_params_degree():
    params = copy.deepcopy(DEF_PARAMS)
    rnd = random.Random()
    rnd.seed(10)
    #seeds = [ rnd.randint(0,65536) for i in xrange(DEGREE_NUM_SAMPLES_PER_PARAM) ]
    #seeds = [ i for i in xrange(DEGREE_NUM_SAMPLES_PER_PARAM) ]

    output_params = S.OutputParams()
    output_params.SKIP_PLOT = True

    for params.STRESS_SAMPLE in ['global', 'semilocal']:
        for params.FLOOR_PLAN_FN in [None, "space.xml"]:
            for params.NOISE_STD, params.PERTURB in [(1e-3, 20), (1e-4, 80)]:
                for params.MAX_DEGREE in xrange(5, 11):
                    for params.RANDOM_SEED in [getCounter() for i in xrange(DEGREE_NUM_SAMPLES_PER_PARAM)]:
                        yield (copy.deepcopy(params), output_params, (0, 0))

def simulation_params_size():
    params = copy.deepcopy(DEF_PARAMS)
    rnd = random.Random()

    rnd.seed(30)
    #sizes = [ (rnd.randint(50, 400), rnd.randint(0, 65536)) for i in xrange(SIZE_NUM_SAMPLES_PER_PARAM) ]
    sizes = [ rnd.randint(50, 400) for i in xrange(SIZE_NUM_SAMPLES_PER_PARAM) ]
    #sizes.extend([ (rnd.randint(400, 600), rnd.randint(0, 65536)) for i in xrange(20) ])
    sizes.sort(lambda x, y: cmp(x[0], y[0]))


    output_params = S.OutputParams()
    output_params.SKIP_PLOT = True

    for params.STRESS_SAMPLE in ['global', 'semilocal']:
        for params.FLOOR_PLAN_FN, params.MAX_DEGREE in [(None, 6), ("space.xml",8)]:
            for params.NOISE_STD, params.PERTURB in [(1e-3, 20)]:#, (1e-4, 80)]:
                for params.V in sizes:
                    params.RANDOM_SEED = getCounter()
                    yield (copy.deepcopy(params), output_params, (0, 0))
    

THRESHOLD = 1.5

def simulate_wrapper(item):
    p = item[0]
    besterror = 1e100
    bestresult = None
    bestparam = None
    if item[1].GENERATE_TXT_GRAPH: 
        newp = copy.deepcopy(p)
        result = dist.simulate(newp, item[1], item[2][0], item[2][1])
        return (newp, result)
    else:
        for i in xrange(1,3):
            newp = copy.deepcopy(p)
            newp.SDP_SAMPLE_MAX = i * p.SDP_SAMPLE_MAX
            result = dist.simulate(newp, item[1], item[2][0], item[2][1])
    
            # Correct a bug that occurs when there's one LCS
            #import datetime 
            #threshold = datetime.datetime.now() - datetime.timedelta(days=2)
            #if 'mtime' in dir(result) and result.mtime < threshold:
            #    if result.n_lcs == 1:
            #        result = dist.simulate(newp, item[1], True)
            #else:
            #    if result.n_lcs == 1:
            #        print "Newly fixed result. No need to regenerate."
    
            if besterror > result.pos_error:
                besterror = result.pos_error
                bestresult = result
                bestparam = newp
            if result.pos_error * newp.meter_ratio() < THRESHOLD:
                break

    # Convert result to real-world units
    r = bestparam.meter_ratio()
    bestresult.pos_error *= r
    bestresult.dis_error *= r
    bestresult.llc_pos_error *= r
    bestresult.llc_dis_error *= r

    f = dir(bestresult)
    if 'edgelen_min' in f:
        bestresult.edgelen_min *= r
        bestresult.edgelen_med *= r
        bestresult.edgelen_max *= r

    return (bestparam, bestresult) 

def scan_for_error(sims, threshold):
    for param, result in sims:
        if result.pos_error > threshold:
            print "Large error: %s\t:%f" % (get_settings_hash(param), result.pos_error)

def go(sim_params, num_worker):
    if num_workers > 1:
        pool = Pool(num_worker)
        return pool.map(simulate_wrapper, sim_params, 1)
    else:
        return map(simulate_wrapper, sim_params)


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

def extract_stats(sims, xname, yname, yscale = 1.0):
    r =  [(getattr(result, xname), yscale * getattr(result, yname)) for params, result in sims]
    r.sort(lambda a, b: cmp(a[0], b[0]))
    return r

def cluster(p):
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

    return (map(lambda (ix, xys): array(xys[1]), h),
            map(lambda (ix, xys): array(xys[0]).mean(), h),
            map(lambda (ix, xys): array(xys[0]).std(), h))

def movavg(x, y, newx, window):
    left = 0
    right = 1
    def weight(x):
        return min(window, max(0, window - x))
    
    avg = zeros((len(newx)), 'd')
    for i in xrange(len(newx)):
        nx = newx[i]
        while right < len(x) and x[right] <= nx+window:
            right = right + 1
        while left < len(x) and x[left] < nx-window:
            left = left + 1

        if left == right: # not enough point
            if left > 0:
                left = left - 1
            if right < len(x):
                right = right + 1

        idx = arange(left, right, dtype='i')
        w = array(map(weight, abs(x[idx]-nx)))
        avg[i] = (y[idx] * w).sum() / w.sum()
    return avg

def myscatter(p, c, sym, ls, smoothwindow):
    x, y = array(map(lambda q:q[0], p)), array(map(lambda q:q[1], p))
    
    r = P.scatter(x, y, s = 15,  marker = sym, alpha=0.7, c=c, linewidth=(0.5,), facecolors=c, edgecolors=c, label=None)

    xp = linspace(x.min(), x.max(), 300)
    avg = movavg(x, y, xp, smoothwindow)
    return P.plot(xp, avg, c = c, linestyle=ls, linewidth=2)
    
def myboxplot(p, sym, c, adj):
    x, positions, xstd = cluster(p)
    r = P.boxplot(x, notch = 0, positions = array(positions) + adj, sym=sym, widths = 0.5)
    P.setp(r['boxes'], color = c, linewidth=1.5)
    P.setp(r['medians'], color = c, linewidth=1.5)
    P.setp(r['whiskers'], color = c, linewidth=1.5)
    P.setp(r['caps'], color = c, linewidth=1.5)
    P.setp(r['fliers'], mfc = None)


def plot_sims_size(sims):
    for floor_plan, max_degree, typ in [(None, 6, 'iso'), ("space.xml",8, 'aniso')]:
        s1 = filter_by_params(sims,
                              STRESS_SAMPLE=['global'],
                              FLOOR_PLAN_FN=[floor_plan])
        s2 = filter_by_params(sims,
                              STRESS_SAMPLE=['semilocal'],
                              FLOOR_PLAN_FN=[floor_plan])

        g_pos = extract_stats(s1, 'v', 'pos_error')
        l_pos = extract_stats(s2, 'v', 'pos_error')
        
        g_llc_pos = extract_stats(s1, 'v', 'llc_pos_error')
        l_llc_pos = extract_stats(s2, 'v', 'llc_pos_error')
        
        g_ratio = extract_stats(s1, 'v', 'v_ratio_llc', yscale=100)
        l_ratio = extract_stats(s2, 'v', 'v_ratio_llc', yscale=100)
        tri_ratio = extract_stats(s1, 'v', 'v_ratio_ltri', yscale=100)

        edge_med = extract_stats(s1, 'v', 'edgelen_med')

        deg = extract_stats(s1, 'v', 'avg_degree')

        g_ns = extract_stats(s1, 'v', 'n_samples')
        l_ns = extract_stats(s2, 'v', 'n_samples')

        P.figure(figsize=(5,4))
        P.axes([0.15,0.12,0.82,0.85])
        a = myscatter(g_pos, c = 'red', sym='^', smoothwindow = 80, ls = '-')
        b = myscatter(l_pos, c = 'blue', sym='s', smoothwindow = 80, ls = '--')

        P.legend([a,b],[r'$\sigma_n$', r'$\sigma_r$'], loc = 4)
            
        P.xlabel("Number of vertices")
        P.ylabel("Positional error (meter)")
        P.yticks([1e-2,1e-1,1e0,1e1])
        P.semilogy()

        P.grid(True)
        P.savefig("%s-versus-v-plot.eps" % typ)


        P.figure(figsize=(5,4))
        P.axes([0.15,0.12,0.82,0.85])
        a = myscatter(g_ratio, c = 'red', sym='^', smoothwindow = 80, ls = '-')
        b = myscatter(l_ratio, c = 'blue', sym='s', smoothwindow = 80, ls = '--')
        c = myscatter(tri_ratio, c = 'green', sym='o', smoothwindow = 80, ls = '-.')

        P.legend([a,b,c],[r'$L_n$', r'$L_r$', r'$L_t$'], loc = 4)
            
        P.xlabel("Number of vertices")
        P.ylabel("Localizable ratio (Percentage)")
        P.yticks(xrange(0,101,10))
        P.ylim(0,110)

        P.grid(True)
        P.savefig("%s-versus-v-ratioplot.eps" % typ)

        P.figure(figsize=(5,4))
        P.axes([0.15,0.12,0.82,0.85])
        a = myscatter(g_ns, c = 'red', sym='^', smoothwindow = 80, ls = '-')
        b = myscatter(l_ns, c = 'blue', sym='s', smoothwindow = 80, ls = '--')

        P.legend([a,b],[r'w/o measurement red.', r'with measurement red.', ], loc = 0)
            
        P.xlabel("Number of vertices")
        P.ylabel("Number of measurements")

        P.grid(True)
        P.savefig("%s-versus-v-nsplot.eps" % typ)


        P.figure(figsize=(5,4))
        myscatter(edge_med, c='red', sym='^', smoothwindow = 80, ls='-')
        P.savefig("%s-edgelen-med-plot.eps" % typ)

        P.figure(figsize=(5,4))
        myscatter(deg, c='red', sym='^', smoothwindow = 80, ls='-')
        P.savefig("%s-avg-degree-plot.eps" % typ)
                              
    
def plot_sims_degree(sims):
    """
    X coordinates is always mean degree
    
    Effects of reduced measurement:
   
    for Type in { iso | aniso}
    for Noise in { Low | High }
    Fig: {global|semilocal}{llc_|}pos_error, {llc_|}dis_error
    Fig: {global|semilocal} v_ratio_llc, v_ratio_ltri
    """

    for method, floor_plan_fn in [('iso', None), ('aniso', "space.xml")]:

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
            # pos_error
            g_pos = extract_stats(s1, 'avg_degree', 'pos_error')
            l_pos = extract_stats(s2, 'avg_degree', 'pos_error')

            #llc_pos_error
            g_llc_pos = extract_stats(s1, 'avg_degree', 'llc_pos_error')
            l_llc_pos = extract_stats(s2, 'avg_degree', 'llc_pos_error')

            #ratio
            g_ratio = extract_stats(s1, 'avg_degree', 'v_ratio_llc', yscale=100)
            l_ratio = extract_stats(s2, 'avg_degree', 'v_ratio_llc', yscale=100)
            tri_ratio = extract_stats(s1, 'avg_degree', 'v_ratio_ltri', yscale=100)



            P.figure(figsize=(5,4))
            P.axes([0.15,0.12,0.84,0.85])
            P.semilogy()

            myboxplot(g_pos, sym = '', c='black', adj = 0.00)
            #myboxplot(l_pos, sym = '', c='blue', adj = -0.00)
            a = myscatter(g_pos, c = 'red', sym='^', smoothwindow = 1, ls = '-')
            b = myscatter(l_pos, c = 'blue', sym='s', smoothwindow = 1, ls = '--')

            xs = array(map(lambda q:q[0], g_pos))
            xt = xrange(int(math.floor(xs.min())),
                        int(math.ceil(xs.max()+0.5)))
            #xt = xrange(4, 11)
            P.xticks(xt)
            P.yticks([1e-2,1e-1,1e0,1e1])

            P.legend([a,b],[r'$\sigma_n$', r'$\sigma_r$'], loc = 4)
            
            P.xlabel("Average vertex degree")
            P.ylabel("Positional error (meter)")
            
            P.grid(True)


            P.savefig("%s-%s-plot.eps" % (method, noise))

            if noise == 'low':
                P.figure(figsize=(5,4))
                P.axes([0.15,0.12,0.83,0.83])

                #myboxplot(g_ratio, sym = '', c='red', adj = 0.00)
                #myboxplot(l_ratio, sym = '', c='blue', adj = 0)
                #myboxplot(tri_ratio, sym = '', c='green', adj = 0.00)

                a = myscatter(g_ratio, c = 'red', sym='^', smoothwindow = 1, ls = '-')
                b = myscatter(l_ratio, c = 'blue', sym='s', smoothwindow = 1, ls = '--')
                c = myscatter(tri_ratio, c='green', sym='o', smoothwindow = 1, ls = '-.')
                
                P.xticks(xt)
                P.yticks(xrange(0,101,10))

                P.xlabel("Average vertex degree")
                P.ylabel("Localizable ratio (percentage)")
                P.grid(True)
                P.ylim(0,110)
                P.legend([a,b,c], [r'$L_n$', r'$L_r$', r'$L_t$'], loc=4)
                
                P.savefig("%s-ratioplot.eps" % (method))



def plot_sdp_degree(fn, outfn):
    f = open(fn,'rb')
    data = []
    while True:
        s = f.readline()
        fields = s.split()
        if (len(fields) != 3):
            break
        data.append((float(fields[0]), float(fields[2])))
    f.close()
    data.sort(lambda a, b: cmp(a[0], b[0]))
    print 'Loaded %d datapoints' % len(data)

    P.figure(figsize=(5,4))
    P.axes([0.15,0.12,0.84,0.85])
    P.semilogy()

    myboxplot(data, sym = '', c='black', adj = 0.00)
            #myboxplot(l_pos, sym = '', c='blue', adj = -0.00)
    a = myscatter(data, c = 'red', sym='^', smoothwindow = 1, ls = '-')

    xs = array(map(lambda q:q[0], data))
    xt = xrange(int(math.floor(xs.min())),
                int(math.ceil(xs.max()+0.5)))
            #xt = xrange(4, 11)
    P.xticks(xt)
    P.yticks([1e-2,1e-1,1e0,1e1])

    P.legend([a],[r'$\sigma_s$'], loc = 4)
            
    P.xlabel("Average vertex degree")
    P.ylabel("Positional error (meter)")
            
    P.grid(True)

    P.savefig(outfn)



def print_graph_txt_list(sims, fn):
    f = open(fn, "wb")
    for param, graph_txt in sims:
        f.write(graph_txt + "\n")
    f.close()

if __name__ == "__main__":
    num_workers = 4
    if len(sys.argv) > 1:
        num_workers = int(sys.argv[1])

    spd = list(simulation_params_degree())
    sps = list(simulation_params_size())

    if (S.OutputParams.GENERATE_TXT_GRAPH):
        if (len(sys.argv) <= 2):
            print "Generate graphs and dump into plain txt"
            sims_d = list(go(spd, num_workers))
            data = filter_by_params(sims_d, 
                                    STRESS_SAMPLE=['global'],
                                    FLOOR_PLAN_FN=['space.xml'],
                                    NOISE_STD=[1e-4])
            data2 = filter_by_params(sims_d, 
                                     STRESS_SAMPLE=['global'],
                                     FLOOR_PLAN_FN=[None],
                                     NOISE_STD=[1e-4])
            print_graph_txt_list(data, "aniso_graphs_by_degree.txt")

            print_graph_txt_list(data2, "iso_graphs_by_degree.txt")
        else:
            plot_sdp_degree("aniso_error.txt", "aniso-sdp-plot.eps")
            plot_sdp_degree("iso_error.txt", "iso-sdp-plot.eps")
    else:

        print "Plot versus degree"
        
        sims_d = list(go(spd, num_workers))
        scan_for_error(sims_d, 1.5) # notify error larger than 1.5 meter
        plot_sims_degree(sims_d)
        
        print "Plot versus size"
        
        sims_s = list(go(sps, num_workers))
        scan_for_error(sims_s, 1.5) # notify error larger than 1.5 meter
        plot_sims_size(sims_s)
        
