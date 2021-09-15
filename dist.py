#!/usr/bin/python
import scipy.stats
import cPickle

import settings as S
from geo import *
from util import *
from graph import *
from tri import *
from plot import *
from genrig import *
import stress
import substress
from timer import Timer
import pickle

from numpy import *
import string


from mitquest import Floorplan, load_floorplan
from scipy.linalg.basic import *
from scipy.linalg.decomp import *

def build_edgeset(p, max_dist, min_dist, max_degree, pred):
    """
    Given a d x n numpyarray p encoding n d-dim points, returns a k x
    2 integer numpyarray encoding k edges whose two ends are distanced
    less than max_dist apart, and each vertex has degree <= max_degree.
    """
    def inbetween (v, l0, l1):
        return v >= l0 and v <= l1

    print_info("Building edge set")
    V = xrange(p.shape[1])
    E = set([])
    deg = zeros(p.shape[1])
    for i in V:
        t = [(norm(p[:,i] - p[:,j]), j) for j in V if i != j]
        t.sort(key = lambda x: x[0])

        for j,(d,k) in enumerate(t):
            if deg[i] >= max_degree:
                break
            
            if inbetween(d, min_dist, max_dist) and pred(p[:,i], p[:,k]):
                edge = (min(i, k), max(i, k))
                if not (edge in E) and deg[i] < max_degree and deg[k] < max_degree:
                    deg[i] = deg[i] + 1
                    deg[k] = deg[k] + 1
                    E.add(edge)

    print_info("\t#e = %d" % len(E))
    return array(list(E), 'i')


def random_graph(params, ignore_cache = False, graph_override = None):
    print_info("Create random graph...")

    v = params.V
    d = params.D
    max_dist = params.dist_threshold()
    min_dist = params.dist_threshold() * 0.05
    max_degree = params.MAX_DEGREE
    
    param_hash = hash((v, d, max_dist, min_dist, max_degree, params.FLOOR_PLAN_FN, params.SINGLE_LC, params.RANDOM_SEED))

    if graph_override == None:
        cache_fn = "%s/graph-%d.cache" % (S.DIR_CACHE, param_hash)
    else:
        cache_fn = graph_override

    def load():
        if ignore_cache:
            return None
        try:
            f = open(cache_fn, "rb")
            g = cPickle.load(f)
            print_info("\tRead from graph cache %s" % cache_fn)
            f.close()
            return g
        except IOError:
            print_info("\tError reading from graph cache %s. Generating graph " % cache_fn)
            return None

    g = load()

    if g == None:
        if params.FLOOR_PLAN_FN:
            floor_plan = load_floorplan("%s/%s" % (S.DIR_DATA, params.FLOOR_PLAN_FN))
            vpred = lambda p: floor_plan.inside(p)
            epred = lambda p, q: not floor_plan.intersect(p, q)
        else:
            floor_plan = None
            vpred = lambda p: 1
            epred = lambda p, q: 1

        p = random_p(v, d, vpred)
        E = build_edgeset(p, max_dist, min_dist, max_degree, epred)
        g = Graph(p, E)
        print_info("|V|=%d |E|=%d" % (g.v, g.e))

        
        #print_info("Filtering dangling vertices...")
        #while 1:
        #    oldv = g.v
        #    g = filter_dangling_v(g)
        #    if oldv == g.v:
        #        break
        #    print_info("\t|V|=%d |E|=%d" % (g.v, g.e))

        g.gr = GenericRigidity(g.v, g.d, g.E)

        if params.SINGLE_LC:

            while True:
                print_info("Extracting largest linked component...")
                lcs = stress.detect_LC_from_kernel(g, g.gr.K_basis)
                largest_lc_id = argmax(array(map(lambda lc: len(lc), lcs), 'i'))
                g = subgraph(g, lcs[largest_lc_id])

                g.gr = GenericRigidity(g.v, g.d, g.E)
                if g.gr.type == 'G':
                    break

        g.floor_plan = floor_plan

        # Calculate results from trilateration

        print_info("Finding largest trilateration graph...")
        g.tri = trilaterate_graph(g.adj)
        print_info("\tsize = %d" % len(g.tri.localized))
        
        f = open(cache_fn, "wb")
        cPickle.dump(g, f)
        f.close()
        print_info("\tWrite to graph cache %s" % cache_fn)
        
    a = g.adj
    md = array(map(len, a), 'd').mean()
    print_info("\t|V|=%d\n\t|E|=%d\n\tMeanDegree=%g (%g)" % (g.v, g.e, g.mean_degree(), md))
    print_info('\ttype = %s\n\trigidity matrix rank = %d  (max = %d)\n\tstress kernel dim = %d (min = %d)'
               % (g.gr.type, g.gr.dim_T, locally_rigid_rank(g.v, d), g.gr.dim_K, d + 1))

    g.hash = param_hash         # store the hash

    return g

def measure_L(g, perturb, noise_std, n_samples, mult_noise):
    print_info("#measurements = %d" % n_samples)

    Ls = asmatrix(zeros((g.e, n_samples), 'd'))
    for i in xrange(n_samples):
        delta = asmatrix(random.uniform(-perturb, perturb, (g.d, g.v)))
        Ls[:,i] = L_map(g.p + delta, g.E, noise_std, mult_noise)

    return Ls, L_map(g.p, g.E, noise_std, mult_noise), L_map(g.p, g.E, 0, mult_noise)

NS = 0

class Conf:
    pass

def dump_graph(g, lc, T_q, fn):
    if (T_q.shape[0]  == 2 or T_q.shape[0] == 3) and len(lc) > 3:
        # dump the graph
        ff = open(fn, "wb")

        sg = subgraph(g, lc)
        conf = Conf()
        if T_q.shape[0] == 2:
            conf.p = vstack((T_q, zeros((1, sg.v))))
        else:
            conf.p = T_q

        conf.v = sg.v
        conf.E = sg.E
        conf.d = d

        pickle.dump(conf, ff)
        ff.close()


class Stats:
    pass

def graph_scale(g, perturb, noise_std, params, output_params, graph_override):
    tm = Timer()
    
    v, p, d, e, E = g.v, g.p, g.d, g.e, g.E
    dim_T = g.gr.dim_T

    tang_var = None
    stress_var = None

    if params.STRESS_SAMPLE == 'semilocal':
        tm.restart()
        Vs, Es = g.get_k_ring_subgraphs(params.K_RING, params.MIN_LOCAL_NBHD)
        print_info("TIME (get_k_ring_subgraphs) = %gs" % tm.elapsed())

        tm.restart()
        g.gsr = GenericSubstressRigidity(g, Vs, Es, params)
        print_info("TIME (Generic substress rigidity) = %gs" % tm.elapsed())

        tm.restart()
        lcs = stress.detect_LC_from_kernel(g, g.gsr.K_basis)
        print_info("TIME (detect LC from kernel) = %gs" % tm.elapsed())
    else:
        tm.restart()
        lcs = stress.detect_LC_from_kernel(g, g.gr.K_basis)
        print_info("TIME (detect LC from kernel) = %gs" % tm.elapsed())
    g.lcs = lcs
    

    if params.STRESS_SAMPLE == 'global':
        n_samples = int(dim_T * params.SAMPLINGS)

        tm.restart()
        Ls, L, exactL = measure_L(g, perturb, noise_std, n_samples, params.MULT_NOISE)
        print_info("TIME (measure_L) = %gs" % tm.elapsed())

        tm.restart()
        if params.EXACT_STRESS:
            S_basis, tang_var = stress.calculate_exact_space(g)
        else:
            S_basis, tang_var = stress.estimate_space(Ls, dim_T)
        print_info("TIME (estimate/calculate stress) = %gs" % tm.elapsed())

        tm.restart()
        ss = stress.sample(S_basis, params) # get stress samples
        print_info("TIME (sample stress) = %gs" % tm.elapsed())
        
    else:
        n_samples = int(max([len(vi) * d for vi in Vs]) * params.SAMPLINGS)
        tm.restart()
        Ls, L, exactL = measure_L(g, perturb, noise_std, n_samples, params.MULT_NOISE)
        print_info("TIME (measure_L) = %gs" % tm.elapsed())

        
        tm.restart()
        sub_S_basis, sparse_param = substress.estimate_space_from_subgraphs(
            Ls = Ls,
            g = g,
            Vs = Vs,
            Es = Es,
            params = params)
        print_info("TIME (estimate_space_from_subgraphs) = %gs" % tm.elapsed())

        if params.STRESS_SAMPLE == 'semilocal':
            tm.restart()
            S_basis, stress_var = substress.consolidate(dim_T, sub_S_basis, sparse_param, params)
            print_info("TIME (consolidate) = %gs" % tm.elapsed())

            tm.restart()
            ss = stress.sample(S_basis, params)
            print_info("TIME (stress.sample) = %gs" % tm.elapsed())

        elif params.STRESS_SAMPLE == 'local':
            stress_var = zeros((1))
            tm.restart()
            ss = substress.sample(sub_S_basis, sparse_param, params)
            print_info("TIME (substress.sample) = %gs" % tm.elapsed())


    

    tm.restart()
    kern = stress.sample_kernel(g, ss)
    print_info("TIME (sample kernel) = %gs" % tm.elapsed())

    approx = asmatrix(zeros((g.d, g.v), 'd'))


    tm_subg = 0
    tm_pca = 0
    tm_sdp = 0
    print_info("Optimizing and fitting each linked components:")
    lcs.reverse()
    for i, lc in enumerate(lcs):
        tm.restart()
        sub_E, sub_E_idx = g.subgraph_edges(lc)
        tm_subg += tm.elapsed()
        
        tm.restart()

        sub_K, s = kern.extract_sub(lc, g.d * params.SDP_SAMPLE_MAX)
        tm_pca = tm_pca + tm.elapsed()

        tm.restart()
        if params.SDP_SAMPLE_MAX == 0:
            q = asmatrix(sub_K)[:, :g.d].T
            if q.shape[0] < g.d:
                q = vstack((q, zeros((g.d-q.shape[0], q.shape[1]))))

            T, Tf = optimal_linear_transform_for_l_lsq(q, d, sub_E, L[sub_E_idx])
        else:
            # enumerate through possible sdp_sample:
            min_error = 1e10
            best_T = None
            rlc = array(ridx(lc, g.v))

            
            if len(lc) <= params.SDP_SAMPLE_ENUMERATE_THRESHOLD:
                sdps_list =  xrange(int(round(g.d * params.SDP_SAMPLE_MIN)), int(round(g.d * params.SDP_SAMPLE_MAX_ENUMERATE))+1, 2)
            else:
                sdps_list = [int(round(g.d * params.SDP_SAMPLE_MAX))]

            for sdps in sdps_list:
                
                q = asmatrix(sub_K)[:, :sdps].T
                #q = asmatrix(sub_K)[:, [min(2, sub_K.shape[1]-1]].T
                if q.shape[0] < g.d:
                    q = vstack((q, zeros((g.d-q.shape[0], q.shape[1]))))
                print_info("GLC #%d:" % (i+1))
                #print_info("\tv=%d\n\te=%d\n\tcond. no.=%g\n\tev=%s\tq.shape=%s" %
                #           (len(lc),
                #            len(sub_E_idx),
                #            cond,
                #            str(s[:min(len(s),d+1)]),
                #            str(q.shape)))
                print_info("\tSDP on %d coordinate vectors..." % q.shape[0])
                T, Tf = optimal_linear_transform_for_l_sdp(q, d, sub_E, L[sub_E_idx])
                Tf_q = Tf * q
    

                Tf_q -= Tf_q.mean(axis=1)

                # some stupid memory error of python/numpy/scipy
                # forces of me to make a fresh copy of Tf_q
                newq = zeros(Tf_q.shape, 'd')
                newq[:,:] = Tf_q[:,:]
                Tf_q = newq

                uu, _t1, _t2 = svd_conv(Tf_q)
                
                T_q = asmatrix(uu).T * Tf_q
                T = (asmatrix(uu).T * Tf)[:d, :]
                
                if output_params.DUMP_LC:
                    dump_graph(g, lc, T_q, "plot/%d-%d-%d.sub" % (i, len(lc), sdps))

                T_q = T_q[:d,:]
                l_error = norm(sqrt(exactL[sub_E_idx]) - sqrt(L_map(T_q, sub_E, 0, params.MULT_NOISE)))

                print_info("\tl_error=%f" % l_error)
                if l_error < min_error:
                    min_error = l_error
                    best_T = T
                    best_q = q
                    
            T = best_T
            q = best_q

                
        tm_sdp = tm_sdp + tm.elapsed()
        
        T_q = T * q
        R = optimal_rigid(T_q, p[:,lc])
        T_q = (R * homogenous_vectors(T_q))[:d,:]
        approx[:, lc] = T_q

        
    print_info("TIME (get subgraph LC) = %gs" % tm_subg)
    print_info("TIME (pca all LC kernel) = %gs" % tm_pca)
    print_info("TIME (sdp all LC kernel) = %gs" % tm_sdp)

    ## Calculate results from trilateration
    if params.TRILATERATION:
        print_info("Performing trilateration for comparison:")
        tm.restart()
        tri = trilaterate_graph(g.adj, sqrt(exactL).A.ravel())
        print_info("TIME (trilaterate) = %gs" % tm.elapsed())
        tri.p = (optimal_rigid(tri.p[:,tri.localized], p[:,tri.localized]) * homogenous_vectors(tri.p))[:d,:]
        tri_g_error = norm(p[:,tri.localized] - tri.p[:,tri.localized]) 
    else:
        tri = g.tri
        tri_g_error = -1

    print_info("\t#localized vertices = %d = (%f%%).\n\tMean error = %f = %fm" %
               (len(tri.localized), 100 *float(len(tri.localized))/v, tri_g_error, tri_g_error * params.meter_ratio()))
    
    # plot the info
    plot_info(g,
              L_opt_p = approx,
              tri = tri,
              dim_T = dim_T,
              tang_var = tang_var,
              stress_var = stress_var,
              stress_spec = abs(kern.eigval),
              perturb = perturb,
              params = params,
              output_params = output_params,
              graph_override = graph_override)

    error = p - approx
    error_l = sqrt(exactL) - sqrt(L_map(approx, E, 0, params.MULT_NOISE))

    g_error = norm(error) / sqrt(float(v))
    l_error = norm(error_l) /sqrt(float(e))

    llc_id = argmax(array(map(lambda lc: len(lc), lcs), 'i')) #largest linked component id
    llc = lcs[llc_id]
    sub_E, sub_E_idx = g.subgraph_edges(llc)
    
    stats = Stats()
    stats.pos_error = g_error
    stats.dis_error = l_error
    stats.llc_pos_error = norm(error[:, llc]) / sqrt(float(len(llc)))
    stats.llc_dis_error = norm(error_l[sub_E_idx]) / sqrt(float(len(sub_E_idx)))
    stats.n_lcs = len(lcs)
    stats.v_ratio_llc = float(len(llc)) / float(g.v)
    if tri:
        stats.v_ratio_ltri = float(len(tri.localized)) / float(g.v)
    else:
        stats.v_ratio_ltri = -1.0
    
    stats.v = g.v
    stats.d = g.d
    stats.avg_degree =  g.mean_degree()

    stats.n_samples = n_samples
    

    print_info("PositionalError = %f = %fm" % (g_error, g_error * params.meter_ratio()))
    print_info("DistanceError = %f = %fm" % (l_error, l_error * params.meter_ratio()))
    print_info("PositionalError of LLC = %f = %fm" % (stats.llc_pos_error, stats.llc_pos_error * params.meter_ratio()))
    print_info("DistanceError of LLC = %f = %fm" % (stats.llc_dis_error, stats.llc_dis_error * params.meter_ratio()))
    print_info("NumLCS = %d" % stats.n_lcs)
    print_info("LLC ratio = %f%%" % (stats.v_ratio_llc * 100))
    print_info("Trilateration ratio = %f%%" % (stats.v_ratio_ltri * 100))

    global NS
    NS = n_samples

    return stats

def generate_txt_graph(g, meter_ratio):
    base_fn = "graph-%d.txt" % g.hash
    fn = S.DIR_CACHE + "/" + base_fn

    f = open(fn, "wb")
    f.write("%d %d %d %g %g\n" % (g.v, g.d, g.e, g.mean_degree(), meter_ratio))
    l = sqrt(L_map(g.p, g.E, 0, 0))

    for i in xrange(g.e):
        f.write("%d %d %g\n" % (g.E[i,0], g.E[i,1], l[i,0]))

    for i in xrange(g.v):
        for j in xrange(g.d):
            f.write("%g " % (g.p[j,i]))
        f.write("\n")

    f.close()
    print_info("\tPlain text version of graph written to %s" % fn)
            
    return base_fn


    

def simulate(params, output_params, ignore_cache = False, ignore_graph_cache = False, graph_override = None):
    if not ignore_cache:
        stats = check_info(params, output_params, graph_override)
        if stats != None:
            return stats

    dump_settings(params)
 
    random.seed(params.RANDOM_SEED)

    tm = Timer()
    tm.restart()
    g = random_graph(params, ignore_graph_cache, graph_override)

    print_info("TIME (random_graph) = %g" % tm.elapsed())

    if output_params.ENUMERATE_GLC:
        return None

    if output_params.GENERATE_TXT_GRAPH:
        fn = generate_txt_graph(g, params.meter_ratio())
        return fn

    edgelen = sqrt(L_map(g.p, g.E, 0.0, params.MULT_NOISE).A.ravel())
    edgelen_min, edgelen_med, edgelen_max = min(edgelen), median(edgelen), max(edgelen)

    print_info("Realworld ratio: 1 unit = %fm " % params.meter_ratio())
    print_info("Edge length:\n\tmin = %f = %fm\n\tmedian = %f = %fm\n\tmax = %f = %fm" % (
        edgelen_min, edgelen_min * params.meter_ratio(),
        edgelen_med, edgelen_med * params.meter_ratio(),
        edgelen_max, edgelen_max * params.meter_ratio()))
    
    if params.MULT_NOISE:
        noise_std = params.NOISE_STD
        perturb = edgelen_med * params.PERTURB * noise_std
    else:
        noise_std = params.NOISE_STD * params.dist_threshold()
        perturb = params.PERTURB * noise_std

    info = 'Graph scale parameters:'
    info += '\n\tmu = sampling factor = %g'    % params.SAMPLINGS
    info += '\n\tdelta/(R*epsilon) = %g' % params.PERTURB
    info += '\n\tN = %d' % params.SS_SAMPLES
    info += '\n\tD = %d' % int(params.SDP_SAMPLE_MAX * g.d)
    info += '\n\tR = %g' % params.dist_threshold()
    info += '\n\tepsilon = noise level = %g'  % (params.NOISE_STD)
    info += '\n\tepsilon*R = noise stddev = %g = %gm' % (noise_std, noise_std * params.meter_ratio())
    info += '\n\tdelta = perturb radius = %g = %gm'     % (perturb, perturb * params.meter_ratio())
    print_info(info)

    tm_total = Timer()
    tm_total.restart()
    
    stats = graph_scale(g = g, 
                        perturb=max(params.MIN_PERTURB, perturb),
                        noise_std = noise_std,
                        params = params,
                        output_params = output_params,
                        graph_override = graph_override)

    stats.edgelen_min = edgelen_min
    stats.edgelen_med = edgelen_med
    stats.edgelen_max = edgelen_max
    stats.v = g.v

    print_info("TIME (total) = %gs" % tm_total.elapsed())
    

    flush_info(params, stats, graph_override)
    
    return stats

if __name__ == "__main__":
    ignore_cache = False
    if len(sys.argv) > 1:
        ignore_cache = int(sys.argv[1]) > 0
    graph_override = None
    if len(sys.argv) > 2:
        graph_override = sys.argv[2]

    simulate(S.SimParams(), S.OutputParams(), ignore_cache = ignore_cache, graph_override = graph_override)
