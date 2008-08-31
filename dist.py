#!/usr/bin/python
import scipy.stats
import cPickle

# Ubuntu Hardy annoyance: the XML lib I use is deprecated
import sys
sys.path = ['/usr/lib/python%s/site-packages/oldxml' % sys.version[:3]] + sys.path 

import settings

from geo import *
from util import *
from graph import *
from tri import *
from plot import *
from genrig import *
from stress import *
from substress import *
from numpy import *
import string

from mitquest import Floorplan
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


def random_graph(v, d, max_dist, min_dist, max_degree):
    print_info("Create random graph...")
    param_hash = hash((v, d, max_dist, min_dist, max_degree, S.FLOOR_PLAN_FN, S.FILTER_DANGLING, S.SINGLE_LC, S.RANDOM_SEED))
    cache_fn = "%s/graph-%d.cache" % (S.DIR_CACHE, param_hash)
    g = None

    try:
        f = open(cache_fn, "r")
        g = cPickle.load(f)
        print_info("\tRead from graph cache %s" % cache_fn)
        f.close()
    except IOError:
        print_info("\tError reading from graph cache %s. Generating graph " % cache_fn)

    if g == None:
        if S.FLOOR_PLAN_FN:
            floor_plan = Floorplan("%s/%s" % (S.DIR_DATA, S.FLOOR_PLAN_FN))
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
        if S.FILTER_DANGLING:
            print_info("Filtering dangling vertices and components...")
            while 1:
                oldv = g.v
                g = filter_dangling_v(g)
                g = largest_cc(g)
                if oldv == g.v:
                    break
                print_info("\t|V|=%d |E|=%d" % (g.v, g.e))

        g.gr = GenericRigidity(g.v, g.d, g.E)

        if S.SINGLE_LC:
            lcs = detect_linked_components_from_stress_kernel(g, g.gr.K_basis)
            g = subgraph(g, lcs[0])
            g.gr = GenericRigidity(g.v, g.d, g.E)
            if g.gr.type != 'G':
                import os
                os.system("echo %d >> glc-non-ggr" % S.RANDOM_SEED)

        g.floor_plan = floor_plan
        f = open(cache_fn, "w")
        cPickle.dump(g, f)
        f.close()
        print_info("\tWrite to graph cache %s" % cache_fn)
        

    print_info("\t|V|=%d\n\t|E|=%d\n\tMeanDegree=%f" % (g.v, g.e, 2.0 * float(g.e)/float(g.v)))
    print_info('\ttype = %s\n\trigidity matrix rank = %d  (max = %d)\n\tstress kernel dim = %d (min = %d)'
               % (g.gr.type, g.gr.dim_T, locally_rigid_rank(g.v, d), g.gr.dim_K, d + 1))
    return g

def measure_L(g, perturb, noise_std, n_samples):
    print_info("#measurements = %d" % n_samples)

    Ls = asmatrix(zeros((g.e, n_samples), 'd'))
    for i in xrange(n_samples):
        delta = asmatrix(random.uniform(-perturb, perturb, (g.d, g.v)))
        Ls[:,i] = L_map(g.p + delta, g.E, noise_std)

    return Ls, L_map(g.p, g.E, noise_std), L_map(g.p, g.E, 0)

def estimate_stress_space(Ls, dim_T):
    u, s, vh = svd(Ls)              # dense
    return u[:,dim_T:], s

            
def sample_from_stress_space(S_basis):
    n_S = S_basis.shape[1]
    nss = min(n_S, S.SS_SAMPLES)

    print_info("Get random stresses...")

    if S.RANDOM_STRESS:
        ss = asmatrix(S_basis) * asmatrix(random.random((n_S, nss+nss/2)))
        if S.ORTHO_SAMPLES:
            ss = svd(ss)[0]
        return ss[:,:nss]
    else:
        # take the last SS_SAMPLES stress vectors in S_basis
        return S_basis[:,n_S-S.SS_SAMPLES:]

NS = 0

def graph_scale(g, perturb, noise_std):
    v, p, d, e, E = g.v, g.p, g.d, g.e, g.E
    dim_T = g.gr.dim_T


    tang_var = None
    stress_var = None
    if S.STRESS_SAMPLE == 'global':
        n_samples = int(dim_T * S.PARAM_SAMPLINGS)

        Ls, L, exactL = measure_L(g, perturb, noise_std, n_samples)

        S_basis, tang_var = estimate_stress_space(Ls, dim_T)
        ss = sample_from_stress_space(S_basis) # get stress samples
        
    else:
        vtx_indices, edge_indices = g.get_k_ring_subgraphs(S.K_RING, S.MIN_LOCAL_NBHD)

        n_samples = int(max([len(vi) * d for vi in vtx_indices]) * S.PARAM_SAMPLINGS)
        Ls, L, exactL = measure_L(g, perturb, noise_std, n_samples)

        sub_S_basis, sparse_param = estimate_sub_stress_space_from_subgraphs(
            Ls = Ls,
            dim_T = dim_T,
            g = g,
            vtx_indices = vtx_indices,
            edge_indices = edge_indices)

        if S.STRESS_SAMPLE == 'semilocal':
            S_basis, stress_var = consolidate_sub_space(dim_T, sub_S_basis, sparse_param)
            ss = sample_from_stress_space(S_basis)

        elif S.STRESS_SAMPLE == 'local':
            stress_var = zeros((1))
            ss = sample_from_sub_stress_space(sub_S_basis, sparse_param)

    K_basis, stress_spec = sample_stress_kernel(g, ss)

    approx = asmatrix(zeros((g.d, g.v), 'd'))

    lcs = detect_linked_components_from_stress_kernel(g, g.gr.K_basis)
    g.lcs = lcs

    print_info("Optimizing and fitting each linked components:")
    for i, lc in enumerate(lcs):
        sub_E, sub_E_idx = g.subgraph_edges(lc)
        sub_K, s, vh = svd(K_basis[lc,:])

        sdps = round(S.SDP_SAMPLE * math.log(float(len(lc))) / math.log(g.v))
        sdps = min(max(sdps, 1), S.SDP_SAMPLE)

        q = asmatrix(sub_K)[:, :int(g.d * sdps)].T
        
        if len(s) > d:
            cond = s[d-1]/s[d]
        else:
            cond = 1e200

        #q = asmatrix(q)[2:3,:]
        if q.shape[0] < g.d:
            q = vstack((q, zeros((g.d-q.shape[0], q.shape[1]))))
        print_info("GLC #%d:" % (i+1))
        print_info("\tv=%d\n\te=%d\n\tcond. no.=%g\n\tev=%s\tq.shape=%s" %
                   (len(lc),
                    len(sub_E_idx),
                    cond,
                    str(s[:min(len(s),d+1)]),
                    str(q.shape)))

        print_info("\tSDP on %d coordinate vectors..." % q.shape[0])

        if S.SDP_SAMPLE == 0:
            T = optimal_linear_transform_for_l_lsq(q, d, sub_E, L[sub_E_idx])
        else:
            T = optimal_linear_transform_for_l_sdp(q, d, sub_E, L[sub_E_idx])
        T_q = T * q
        R = optimal_rigid(T_q, p[:,lc])
        T_q = (R * homogenous_vectors(T_q))[:d,:]
        approx[:, lc] = T_q

    ## Calculate results from trilateration
    if S.TRILATERATION:
        print_info("Performing trilateration for comparison:")
        tri = trilaterate_graph(g.adj, sqrt(exactL).A.ravel())
        tri.p = (optimal_rigid(tri.p[:,tri.localized], p[:,tri.localized]) * homogenous_vectors(tri.p))[:d,:]
    else:
        tri = None


    g_error = norm(p - approx) / sqrt(float(v))
    l_error = norm(sqrt(exactL) - sqrt(L_map(approx, E, 0))) /sqrt(float(e))

    if tri:
        tri_g_error = norm(p[:,tri.localized] - tri.p[:,tri.localized])
        print_info("\t#localized vertices = %d = (%f%%).\n\tMean error = %f = %fm" %
                   (len(tri.localized), 100 *float(len(tri.localized))/v, tri_g_error, tri_g_error * S.meter_ratio()))
    
    # plot the info
    plot_info(g,
              L_opt_p = approx,
              tri = tri,
              dim_T = dim_T,
              tang_var = tang_var,
              stress_var = stress_var,
              stress_spec = sqrt(stress_spec),
              perturb = perturb)

    print_info("PositionalError = %f = %fm" % (g_error, g_error * S.meter_ratio()))
    print_info("DistanceError = %f = %fm" % (l_error, l_error * S.meter_ratio()))

    global NS
    NS = n_samples

    return g_error, l_error

def simulate(ignore_cache = False):
    if not ignore_cache:
        e = check_info()
        if e != None:
            return e

    dump_settings()

    random.seed(S.RANDOM_SEED)

    g = random_graph(v = S.PARAM_V,
                     d = S.PARAM_D,
                     max_dist = S.dist_threshold(),
                     min_dist = S.dist_threshold() * 0.05,
                     max_degree = S.MAX_DEGREE)

    if S.sENUMERATE_GLC:
        return -1, -1

    edgelen = sqrt(L_map(g.p, g.E, 0.0).A.ravel())
    edgelen_min, edgelen_med, edgelen_max = min(edgelen), median(edgelen), max(edgelen)

    print_info("Realworld ratio: 1 unit = %fm " % S.meter_ratio())
    print_info("Edge length:\n\tmin = %f = %fm\n\tmedian = %f = %fm\n\tmax = %f = %fm" % (
        edgelen_min, edgelen_min * S.meter_ratio(),
        edgelen_med, edgelen_med * S.meter_ratio(),
        edgelen_max, edgelen_max * S.meter_ratio()))
    
    if S.MULT_NOISE:
        noise_std = S.PARAM_NOISE_STD
        perturb = edgelen_med * S.PARAM_PERTURB * noise_std
    else:
        noise_std = S.PARAM_NOISE_STD * S.dist_threshold()
        perturb = S.PARAM_PERTURB * noise_std

    info = 'Graph scale parameters:'
    info += '\n\tmu = sampling factor = %g'    % S.PARAM_SAMPLINGS
    info += '\n\tdelta/(R*epsilon) = %g' % S.PARAM_PERTURB
    info += '\n\tN = %d' % S.SS_SAMPLES
    info += '\n\tD = %d' % int(S.SDP_SAMPLE * g.d)
    info += '\n\tR = %g' % S.dist_threshold()
    info += '\n\tepsilon = noise level = %g'  % (S.PARAM_NOISE_STD)
    info += '\n\tepsilon*R = noise stddev = %g = %gm' % (noise_std, noise_std * S.meter_ratio())
    info += '\n\tdelta = perturb radius = %g = %gm'     % (perturb, perturb * S.meter_ratio())
    print_info(info)

    e = graph_scale(g = g, 
                    perturb=max(S.PARAM_MIN_PERTURB, perturb),
                    noise_std = noise_std)

    flush_info(e[0], e[1])
    return e

if __name__ == "__main__":
    if len(sys.argv) > 1:
        simulate(int(sys.argv[1]) > 0)
    else:
        simulate(False)



