#!/usr/bin/python
import scipy.stats
import cPickle

import sys
sys.path = ['/usr/lib/python%s/site-packages/oldxml' % sys.version[:3]] + sys.path # hardy annoyance

from settings import *
from geo import *
from util import *
from graph import *
from tri import *
from plot import *
from genrig import *
from stress import *
from numpy import *
import string

from mitquest import Floorplan
from scipy.linalg.basic import *
from scipy.linalg.decomp import *

class NotLocallyRigid(Exception):
    pass

class TooFewSamples(Exception):
    pass
    

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
    for i in V:
        t = [(norm(p[:,i] - p[:,j]), j) for j in V if i != j]
        t.sort(key = lambda x: x[0])

        n = 0
        for j,(d,k) in enumerate(t):
            if inbetween(d, min_dist, max_dist) and pred(p[:,i], p[:,k]):
                E.add((min(i, k), max(i, k)))
                n = n + 1
            if n >= max_degree:
                break

    print_info("\t#e = %d" % len(E))
    return array(list(E), 'i')


def random_graph(v, d, max_dist, min_dist, max_degree, vpred, epred):
    print_info("Create random graph...")
    param_hash = hash((v, d, max_dist, min_dist, max_degree, FLOOR_PLAN_FN, FILTER_DANGLING))
    cache_fn = "%s/graph-%d.cache" % (DIR_CACHE, param_hash)
    g = None

    try:
        f = open(cache_fn, "r")
        g = cPickle.load(f)
        print_info("\tRead from graph cache %s" % cache_fn)
        f.close()
    except IOError:
        print_info("\tError reading from graph cache %s. Generating graph " % cache_fn)

    if g == None:
        p = random_p(v, d, vpred)
        E = build_edgeset(p, max_dist, min_dist, max_degree, epred)
        g = Graph(p, E)
        print_info("|V|=%d |E|=%d" % (g.v, g.e))
        if FILTER_DANGLING:
            print_info("Filtering dangling vertices and components...")
            while 1:
                oldv = g.v
                g = filter_dangling_v(g)
                g = largest_cc(g)
                if oldv == g.v:
                    print_info("Done! |V|=%d |E|=%d" % (g.v, g.e))
                    break
                print_info("\t|V|=%d |E|=%d" % (g.v, g.e))

        g.gr = GenericRigidity(g.v, g.d, g.E)

        f = open(cache_fn, "w")
        cPickle.dump(g, f)
        f.close()
        print_info("\tWrite to graph cache %s" % cache_fn)

    print_info("\t|V|=%d |E|=%d" % (g.v, g.e))
    print_info('\ttype = %s\n\trigidity matrix rank = %d  (max = %d)\n\tstress kernel dim = %d (min = %d)'
               % (g.gr.type, g.gr.dim_T, locally_rigid_rank(g.v, d), g.gr.dim_K, d + 1))
    return g

def measure_L(g, perturb, noise_std, n_samples):
    print_info("#measurements = %d" % n_samples)

    Ls = asmatrix(zeros((g.e, n_samples), 'd'))
    for i in xrange(n_samples):
        delta = asmatrix(random.uniform(-perturb, perturb, (g.d, g.v)))
        #for i in xrange(g.v):
        #    delta[:,i] /= norm(delta[:,i])
        #delta *= perturb
        Ls[:,i] = L_map(g.p + delta, g.E, noise_std)

    return Ls, L_map(g.p, g.E, noise_std)

def estimate_stress_space(Ls, dim_T):
    u, s, vh = svd(Ls)              # dense
    S_basis = asmatrix(u[:,dim_T:])
    return S_basis, s

def affected_vertices(E, edge_idx):
    return set(E[edge_idx,:].ravel())

def estimate_sub_stress_space(Ls, g, edge_idx, vtx_idx = None):
    """
    edge_idx is a 1-d array of edge indices. Only the indexed edges
    have the correponding components of the stress space calculated.
    """
    u, s, vh = svd_conv(Ls[edge_idx, :]) # dense

    if vtx_idx != None:
        v = len(vtx_idx)
    else:
        v = len(affected_vertices(g.E, edge_idx))
    d = g.d

    # sanity check for subgraph
    sub_dim_T = locally_rigid_rank(v, d)
    if sub_dim_T > len(edge_idx) or sub_dim_T > Ls.shape[1]:
        raise TooFewSamples()

    # Now find the actual dimension of the tangent space, it should be
    # in the range of [locally_rigid_rank(v, d), v * d]
    prev_derivative = 1                   # derivative should always < 0
    t = sub_dim_T
    while t < min(len(edge_idx), Ls.shape[1], v * d):
        derivative = math.log(s[t] / s[t-1]) / math.log(float(t)/float(t-1))
        if derivative > prev_derivative:
            break
        prev_derivative = derivative
        t = t + 1

    sub_S_basis = asmatrix(u[:,t-1:])
    return sub_S_basis, s, (t-1-sub_dim_T)

# This doesn't work for some reason!
def calculate_exact_sub_stress_space(g, edge_idx, vtx_idx = None):
    p, E = g.p, g.E
    if vtx_idx == None:
        vtx_idx = affected_vertices(E, edge_idx)
    v = len(vtx_idx)

    invert_vtx_idx = -ones((v_d_e_from_graph(g)[0]), 'i')
    for i, w in enumerate(vtx_idx):
        invert_vtx_idx[w] = i

    d = g.d

    D = zeros((len(edge_idx), d * v))
    print D.shape
    for k, ei in enumerate(edge_idx):
        i, j = E[ei][0], E[ei][1]
        diff_ij = p[:,i] - p[:,j]
        i = invert_vtx_idx[i]
        j = invert_vtx_idx[j]
        D[k, i*d:i*d+d] = 2.0 * diff_ij.T
        D[k, j*d:j*d+d] = -2.0 * diff_ij.T

    u, s, vh = svd(D)               # E by dv, sparse, 2dE non-zero
    sub_dim_T =  locally_rigid_rank(v, d)
    t = sub_dim_T
    while t < min(len(edge_idx), v * d):
        if s[t] < EPS:
            break
        t = t + 1

    return u[:,t:], s, (t-sub_dim_T)
    


def estimate_sub_stress_space_from_subgraphs(Ls, dim_T, g, vtx_indices, edge_indices):
    v, e = g.v, g.e

    print_info("Computing stress space for each subgraph")
    sub_S_basis = []
    n = 0
    nz = 0
    missing_stress = []
    for i in xrange(v):

        if EXACT_LOCAL_STRESS:
            result = calculate_exact_sub_stress_space(g, edge_indices[i], vtx_indices[i])
        else:
            result = estimate_sub_stress_space(Ls, g, edge_indices[i], vtx_indices[i])

        sub_S_basis.append(result[0])
        missing_stress.append(result[2])
        
        n = n + sub_S_basis[i].shape[1]
        nz = nz + sub_S_basis[i].shape[0] * sub_S_basis[i].shape[1]
        sys.stdout.write('.')
        sys.stdout.flush()
    sys.stdout.write('\n')

    missing_stress = array(missing_stress, 'd')
    print_info("Local stress space lost dim:\n\tavg = %f\n\tmax = %f" % (
        mean(missing_stress), max(missing_stress)))

    return sub_S_basis, (e, n, nz, edge_indices)

def consolidate_sub_space(dim_T, sub_basis, sparse_param):
    e, n, nz, edge_indices = sparse_param

    print_info("Consolidating subspaces...")

    if USE_SPARSE_SVD:
        read_succ = False
        if CONSOLIDATE_READ_FROM_CACHE:
            try:
                fn = "%s/consolidate-pca-%d-%d-%d.cache" % (DIR_CACHE, e, n, nz)
                f = open(fn, "r")
                u, s = cPickle.load(f)
                print_info("\tRead from consolidation PCA cache %s" % fn)
                read_succ = True
            except IOError:
                print_info("\tError reading from consolidation PCA cache %s. Perform PCA..." % fn)

        if (not CONSOLIDATE_READ_FROM_CACHE) or (not read_succ):
            # Now write a temporary file of the big sparse matrix
            print_info("Write out %dx%d sparse matrix for external SVDing" % (e, n) )
            f = open("%s/input.st" % DIR_TMP, "w")
            f.write("%d %d %d\n" % (e, n, nz))
            for k, B in enumerate(sub_basis):
                for i in xrange(B.shape[1]):
                    f.write("%d" % B.shape[0])
                    for j in xrange(B.shape[0]):
                        f.write(" %d %f" % (edge_indices[k][j], B[j,i]))
                    f.write("\n")
            f.close()

            # and use './svd' to svd factorize it
            import os
            os.system("%s/svd %s/input.st -o %s/output -d %d " % (DIR_BIN, DIR_TMP, DIR_TMP, e-dim_T))

            print_info("Read back SVD result")
            f = open("%s/output-Ut" % DIR_TMP, "r")      # read in columns
            f.readline()                        # skip first line (dimension info)
            u = zeros((e, e-dim_T), "d")
            for i in xrange(e-dim_T):           # now grab each column
                toks = string.split(f.readline())
                c = array([map(float, toks)]).T
                u[:,i] = c[:,0]
            f.close()

            f = open("%s/output-S" % DIR_TMP, "r")           # read in singular values
            f.readline()
            s = array(map(float, string.split(f.read())))
            f.close()

            if CONSOLIDATE_WRITE_TO_CACHE:
                fn = "%s/consolidate-pca-%d-%d-%d.cache" % (DIR_CACHE, e, n, nz)
                f = open(fn, "w")
                cPickle.dump((u, s),f)
                f.close()
                print_info("\tWrite to consolidation PCA cache %s" % fn)
    else:
    
        #The following uses scipy dense matrix routines

        # Now unpack the various sub_basis into one big matrix
        m = zeros((e, n))
        j = 0;
        for i in xrange(v):
            j2 = j + sub_basis[i].shape[1]
            m[edge_indices[i], j:j2] = sub_basis[i]
            j = j2

        # The matrix is e by n, where
        # n = v * avg_local_stress_dim
        #   = v * (avg_local_n_edges - avg_local_n_v * d)
        #   = v * avg_local_n_v * (avg_degree - d)
        #   = ~ 4v * avg_local_n_v
        # since avg_degree = 6, and d = 2
        #
        # The number of non-zero entries are
        # n * avg_local_n_edges
        # = 4v * avg_local_n_v^2 * avg_degree
        # = 24v * avg_local_n_v^2
        #
        # avg_local_n_v = ~30 for 2-ring, ~60 for 3-ring, ~100 for
        # 4-ring

        u, s, vh = svd(m)
        u = u[:, :(e-dim_T)], s[:e-dimT]

    thr = median(s) * STRESS_VAL_PERC /100
    i = e-dim_T-1
    while  i >= 0 and s[i] < thr:
        i = i-1
    print_info("Top %d (%f%%) of %d stresses used" % (i+1, 100*float(i+1)/(e-dim_T), e-dim_T))
    return u[:,:i+1], s

def sample_from_sub_stress_space(sub_S_basis, sparse_param):
    e, n, nz, edge_indices = sparse_param

    print_info("Sampling from sub stress spaces...")
    stress_samples = zeros((e, SS_SAMPLES+SS_SAMPLES/2), order = 'FORTRAN')
    for i in xrange(SS_SAMPLES+SS_SAMPLES/2):
        for j, basis in enumerate(sub_S_basis):
            w = asmatrix(basis) * asmatrix(random.random((basis.shape[1], 1)))
            w /= norm(w)
            stress_samples[edge_indices[j],i:i+1] += w

    if ORTHO_SAMPLES:
        stress_samples = svd(stress_samples)[0][:,:stress_samples.shape[1]]

    return stress_samples[:,:SS_SAMPLES]            
            
def sample_from_stress_space(S_basis):
    n_S = S_basis.shape[1]

    print_info("Sampling from stress space...")

    # If you want randomize:
    stress_samples = asmatrix(S_basis) * asmatrix(random.random((n_S, SS_SAMPLES+SS_SAMPLES/2)))

    # Else take the last ss_samples stress vectors in S_basis
    #stress_samples = S_basis[:,n_S-SS_SAMPLES:]

    if ORTHO_SAMPLES:
        stress_samples = svd(stress_samples)[0][:,:stress_samples.shape[1]]

    return stress_samples[:,:SS_SAMPLES]
    #return S_basis

def graph_scale(g, perturb, noise_std, sampling, k, min_neighbor, floor_plan):
    v, p, d, e, E = g.v, g.p, g.d, g.e, g.E

    info = 'Graph scale parameters:'
    info += '\n\tv = %d'        % v
    info += '\n\tdimension = %d' % d
    info += '\n\tdist threshold = %1.02f'   % PARAM_DIST_THRESHOLD
    info += '\n\tmax neighbors = %d' % MAX_DEGREE
    info += '\n\tnoise = %.04f (= %.04fm for additive noise)'     % (noise_std, noise_std * METER_RATIO)
    info += '\n\tperturb = %.04f = %.04fm'     % (perturb, perturb * METER_RATIO)
    info += '\n\tsampling = %2.02f'    % sampling
    info += '\n\tkernel samples = %d'     % KERNEL_SAMPLES
    info += '\n\tortho samples = %s'       % str(ORTHO_SAMPLES)[0]
    info += '\n\tmult noise = %s'       % str(MULT_NOISE)[0]           
    info += '\n\texact local stress = %s'       % str(EXACT_LOCAL_STRESS)[0]   
    info += '\n\tstress space samples = %d'     % SS_SAMPLES                   
    info += '\n\tweight kernel samples = %s'      % str(WEIGHT_KERNEL_SAMPLE)[0] 
    info += '\n\tstress value perc = %d%%'    % STRESS_VAL_PERC              
    info += '\n\tstress sample method = %s'         % STRESS_SAMPLE
    info += '\n\tfloor plan = %s'                   % FLOOR_PLAN_FN
    info += '\n\tpca stress kernel per linked comp. = %s' % str(PER_LC_KS_PCA)[0]
    info += '\n\tSDP sample = %d' % SDP_SAMPLE
    info += '\n\tSDP use DSDP = %s' % str(SDP_USE_DSDP)[0]
    print_info(info)


    dim_T = g.gr.dim_T

    if STRESS_SAMPLE == 'global':
        n_samples = int(dim_T * sampling)

        Ls, L = measure_L(g, perturb, noise_std, n_samples)

        S_basis, tang_var = estimate_stress_space(Ls, dim_T)
        stress_samples = sample_from_stress_space(S_basis)
        
    else:
        vtx_indices, edge_indices = g.get_k_ring_subgraphs(k, min_neighbor)

        n_samples = int(max([len(vi) * d for vi in vtx_indices]) * sampling)
        Ls, L = measure_L(g, perturb, noise_std, n_samples)

        sub_S_basis, sparse_param = estimate_sub_stress_space_from_subgraphs(
            Ls = Ls,
            dim_T = dim_T,
            g = g,
            vtx_indices = vtx_indices,
            edge_indices = edge_indices)

        if STRESS_SAMPLE == 'semilocal':
            S_basis, tang_var = consolidate_sub_space(dim_T, sub_S_basis, sparse_param)
            stress_samples = sample_from_stress_space(S_basis)

        elif STRESS_SAMPLE == 'local':
            tang_var = zeros((1))
            stress_samples = sample_from_sub_stress_space(sub_S_basis, sparse_param)

    K_basis, stress_spec = sample_stress_kernel(g, stress_samples)

    l_approx = asmatrix(zeros((g.d, g.v), 'd'))
    af_approx = asmatrix(zeros((g.d, g.v), 'd'))


    lcs = detect_linked_components_from_stress_kernel(g, g.gr.K_basis)

    print_info("Optimizing and fitting each linked components:")
    for lc in lcs:
        sub_E, sub_E_idx = g.subgraph_edges(lc)
        sub_K, s, vh = svd(K_basis[lc,:])

        if SS_SAMPLES == 1:
            q = asmatrix(K_basis)[lc, :int(g.d * SDP_SAMPLE)].T
        else:
            q = asmatrix(sub_K)[:, :int(g.d * SDP_SAMPLE)].T
            if PER_LC_KS_PCA:
                stress_spec = s
        
        if len(s) > d:
            cond = s[d-1]/s[d]
        else:
            cond = 1e200

        #q = asmatrix(q)[2:3,:]
        if q.shape[0] < g.d:
            q = vstack((q, zeros((g.d-q.shape[0], q.shape[1]))))
        print_info("\tv=%d\te=%d\tcond. no.=%g\tev=%s\tq.shape=%s" %
                   (len(lc),
                    len(sub_E_idx),
                    cond,
                    str(s[:min(len(s),d+1)]),
                    str(q.shape)))

        print_info("Performing SDP to recover configuration from %d coordinate vectors..." % q.shape[0])

        if SDP_SAMPLE == 0:
            T = optimal_linear_transform_for_l_lsq(q, d, sub_E, L[sub_E_idx])
        else:
            T = optimal_linear_transform_for_l_sdp(q, d, sub_E, L[sub_E_idx])
        T_q = T * q
        R = optimal_rigid(T_q, p[:,lc])
        T_q = (R * homogenous_vectors(T_q))[:d,:]
        l_approx[:, lc] = T_q

        af_approx[:,lc] = T_q
        
        #tr = optimal_affine(q, p[:,lc])
        #af_approx[:, lc] = (tr *
        #                    homogenous_vectors(q))[:d,:]
        
    

    ## Calculate results from trilateration
    ## print_info("Performing trilateration for comparison:")
    ## tri = trilaterate_graph(g.adj, sqrt(L).A.ravel())
    ## tri.p = (optimal_rigid(tri.p[:,tri.localized], p[:,tri.localized]) * homogenous_vectors(tri.p))[:d,:]
    tri = None

    l = L_map(p, E, 0)

    def mean_l2_error(v1, v2):
        d, n = v1.shape
        return mean(array([norm(v1[:,i] - v2[:,i]) for i in xrange(n)]))
    
    af_g_error = mean_l2_error(p, af_approx)
    af_l_error = mean_l2_error(l.T, L_map(af_approx, E, 0).T)

    l_g_error = mean_l2_error(p, l_approx)
    l_l_error = mean_l2_error(l.T, L_map(l_approx, E, 0).T)

    #tri_g_error = mean_l2_error(p[:,tri.localized], tri.p[:,tri.localized])

    #print("\t#localized vertices = %d = (%f%%).\n\tMean error = %f = %fm" %
    #      (len(tri.localized), 100 *float(len(tri.localized))/v, tri_g_error, tri_g_error * METER_RATIO))
    
    # plot the info
    plot_info(g,
              L_opt_p = l_approx,
              aff_opt_p = af_approx,
              tri = tri,
              dim_T = dim_T,
              tang_var = sqrt(tang_var),
              stress_spec = sqrt(stress_spec),
              floor_plan = floor_plan,
              stats = {"noise":noise_std,
                       "perturb":perturb,
                       "sampling":sampling,
                       "samples":n_samples,
                       "l g error": l_g_error,
                       "l l error": l_l_error,
                       "af g error": af_g_error,
                       "af l error": af_l_error})

    print_info("Mean error %f = %fm" % (l_g_error, l_g_error * METER_RATIO))

    return l_g_error

def main():
    import sys
    random.seed(0)
    k =  math.pow(2*(PARAM_D+1), 1.0/PARAM_D)/3.0
    dist_threshold = PARAM_DIST_THRESHOLD*k*math.pow(PARAM_V, -1.0/PARAM_D)

    if FLOOR_PLAN_FN:
        floor_plan = Floorplan("%s/%s" % (DIR_DATA, FLOOR_PLAN_FN))
        vpred = lambda p: floor_plan.inside(p)
        epred = lambda p, q: not floor_plan.intersect(p, q)
    else:
        floor_plan = None
        vpred = lambda p: 1
        epred = lambda p, q: 1

    g = random_graph(v = PARAM_V,
                     d = PARAM_D,
                     max_dist = dist_threshold,
                     min_dist = dist_threshold * 0.05,
                     max_degree = MAX_DEGREE,
                     vpred = vpred,
                     epred = epred)

    #g = largest_cc(g)
    #g.gr = GenericRigidity(g.v, g.d, g.E)

    if SINGLE_LC:
        lcs = detect_linked_components_from_stress_kernel(g, g.gr.K_basis)
        g = subgraph(g, lcs[0])
        g.gr = GenericRigidity(g.v, g.d, g.E)

    edgelen = sqrt(L_map(g.p, g.E, 0.0).A.ravel())
    edgelen_min, edgelen_med, edgelen_max = min(edgelen), median(edgelen), max(edgelen)
    global METER_RATIO
    METER_RATIO = MAX_EDGE_LEN_IN_METER / edgelen_max
    print_info("Realworld ratio: 1 unit = %fm " % METER_RATIO)
    print_info("Edge length:\n\tmin = %f = %fm\n\tmedian = %f = %fm\n\tmax = %f = %fm" % (
        edgelen_min, edgelen_min * METER_RATIO,
        edgelen_med, edgelen_med * METER_RATIO,
        edgelen_max, edgelen_max * METER_RATIO))
    
    if MULT_NOISE:
        noise_std = PARAM_NOISE_STD
        perturb = edgelen_med * PARAM_PERTURB * noise_std
    else:
        noise_std = PARAM_NOISE_STD * dist_threshold
        perturb = PARAM_PERTURB * noise_std

    e = graph_scale(g = g, 
                    perturb=max(PARAM_MIN_PERTURB, perturb),
                    noise_std = noise_std,
                    sampling = PARAM_SAMPLINGS,
                    k = K_RING,
                    min_neighbor = MIN_LOCAL_NBHD,
                    floor_plan = floor_plan)
    sys.stdout.flush()


if __name__ == "__main__":
    main()
