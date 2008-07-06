#!/usr/bin/python
import scipy.stats
import cPickle

from settings import *
from geo import *
from util import *
from graph import *
from sparselin import *
from tri import *
from plot import *
from genrig import *
from stress import *
from numpy import *
#from floorplan import *
from mitquest import Floorplan
from scipy.linalg.basic import *
from scipy.linalg.decomp import *

class NotLocallyRigid(Exception):
    pass

class TooFewSamples(Exception):
    pass
    

def build_edgeset(p, max_dist, min_dist, max_neighbors = None, floor_plan = None):
    """
    Given a d x n numpyarray p encoding n d-dim points, returns a k x
    2 integer numpyarray encoding k edges whose two ends are distanced
    less than max_dist apart, and each vertex has max number of
    neighbors <= max_neighbors.
    """
    def inbetween (v, l0, l1):
        return v >= l0 and v <= l1

    print_info("Building edge set")

    V = xrange(p.shape[1])

    if max_neighbors == None:
        A = array(
            [[i,j] for i in V for j in V if i < j and
             inbetween(norm(p[:,i] - p[:,j]), min_dist, max_dist) and
             (floor_plan == None or
              not floor_plan.intersect(p[:,i], p[:,j]))], 'i')
    else:
        E = set([])
        for i in V:
            t = [(norm(p[:,i] - p[:,j]), j)
                 for j in V if i != j]
            t.sort(key = lambda x: x[0])
            t = filter(lambda x: inbetween(x[0], min_dist, max_dist), t)

            j = k = 0
            while j < len(t) and k < max_neighbors:
                if inbetween(t[j][0], min_dist, max_dist) and (
                    floor_plan == None or not floor_plan.intersect(p[:,i], p[:,t[j][1]])):
                    E.add((min(i, t[j][1]), max(i,t[j][1])))
                    k = k + 1
                j = j + 1
              
        A = array(list(E), 'i')

    print_info("\t#e = %d" % A.shape[0])
    return A


def random_graph(v, d, max_dist, min_dist, discardNonrigid, max_neighbors, floor_plan):
    i = 0
    while 1:
        p = random_p(v, d, lambda p:floor_plan.inside(p))
        E = build_edgeset(p, max_dist, min_dist, max_neighbors, floor_plan)
        e = len(E)
        i = i + 1
        g = Graph(p, E)

        if discardNonrigid:
            def filter_dangling_v(g):
                while 1:
                    ov = g.v
                    adj = g.adj
                    g = subgraph(g, [i for i in xrange(g.v) if len(adj[i]) > d])
                    if g.v == ov:
                        break
                return g

            def filter_dangling_c(g):
                cc, cn = g.connected_components()
                if len(cc) == 1:
                    return g
                c = argmax(cc)
                return subgraph(g, [i for i in xrange(g.v) if cn[i] == c])

            print_info("|V|=%d |E|=%d" % (g.v, g.e))
            print_info("Filtering dangling vertices and components...")
            while 1:
                oldv = g.v
                g = filter_dangling_v(g)
                g = filter_dangling_c(g)
                if oldv == g.v:
                    print_info("Done! |V|=%d |E|=%d" % (g.v, g.e))
                    break
                print_info("\t|V|=%d |E|=%d" % (g.v, g.e))
            
            gr = GenericRigidity(g.v, g.d, g.E)
            if gr.type == 'N':
                print_info("Not generically locally rigid")
                continue
            g.gr = gr

            print_info("Graph created after %d tries" % i)

        return g

def sparse_stress_matrix_template(g):
    adj = g.adj

    nzval_idx = []
    irow = []
    pcol = [0]
    for i in xrange(g.v):
        nzval_idx.append(-1)
        irow.append(i)
        for e, dest in adj[i]:
            if dest > i:
                nzval_idx.append(e)
                irow.append(dest)
        pcol.append(len(irow))
    return (array(nzval_idx, 'i'),
            [array(map(lambda l: l[0], a), 'i') for a in adj],
            array(irow, 'i'),
            array(pcol,'i'))
        

def stress_matrix_speig_from_vector(w, sparse_template, nzval, nev, eigval, eigvec):
    nzval_idx, adj_idx, irow, pcol = sparse_template
    for i in xrange(1, len(pcol)):
        nzval[pcol[i-1]+1:pcol[i]] = asarray(w[nzval_idx[pcol[i-1]+1:pcol[i]],:].T)
        nzval[pcol[i-1]] = - sum(w[adj_idx[i-1]])

    speig(n=len(pcol)-1,
          nzval=nzval,
          irow=irow,
          pcol=pcol, lu='L',
          nev=nev,
          which='SM',
          eigval=eigval,
          eigvec=eigvec,
          maxit=len(pcol)*100,
          tol=0.5)

    order =  range(nev)
    order.sort(key = lambda i: abs(eigval[i]))
    eigval[:] = eigval[order]
    eigvec[:,:] = eigvec[:,order]

    #print eigval

def check_gr(g):
    if not 'gr' in g.__dict__ or g.gr.type == 'N':
        raise NotLocallyRigid()

def measure_L_rho(g, perturb, noise_std, n_samples):
    print_info("#measurements = %d" % n_samples)

    L_rhos = asmatrix(zeros((g.e, n_samples), 'd'))
    for i in xrange(n_samples):
        delta = asmatrix(random.random((g.d, g.v))) - 0.5
        delta *= (perturb*2)
        L_rhos[:,i] = L(g.p + delta, g.E, noise_std)

    return L_rhos, L(g.p, g.E, noise_std)

def estimate_stress_space(L_rhos, dim_T):
    u, s, vh = svd(L_rhos)              # dense
    S_basis = asmatrix(u[:,dim_T:])
    return S_basis, s

def affected_vertices(E, edge_idx):
    return set(E[edge_idx,:].ravel())

def estimate_sub_stress_space(L_rhos, g, edge_idx, vtx_idx = None):
    """
    edge_idx is a 1-d array of edge indices. Only the indexed edges
    have the correponding components of the stress space calculated.
    """
    u, s, vh = svd_conv(L_rhos[edge_idx, :]) # dense

    if vtx_idx != None:
        v = len(vtx_idx)
    else:
        v = len(affected_vertices(g.E, edge_idx))
    d = g.d

    # sanity check for subgraph
    sub_dim_T = locally_rigid_rank(v, d)
    if sub_dim_T > len(edge_idx) or sub_dim_T > L_rhos.shape[1]:
        raise TooFewSamples()

    # Now find the actual dimension of the tangent space, it should be
    # in the range of [locally_rigid_rank(v, d), v * d]
    prev_derivative = 1                   # derivative should always < 0
    t = sub_dim_T
    while t < min(len(edge_idx), L_rhos.shape[1], v * d):
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
    

def get_k_ring_subgraphs(g, k, min_neighbor):
    """
    return k_rings of each vertex. If the #v of the ring is less than
    min_neighbor, additional vertices in larger rings are added till
    at least min_neighbor of them is in the graph. Result is stored
    as two list, one vtx indices, one edge indices.
    """
    print_info("Computing %d-ring subgraphs" % k)
    v, adj = g.v, g.adj
    vtx_indices = [set([]) for i in xrange(v)]
    edge_indices = [set([]) for i in xrange(v)]
    anv = 0
    ane = 0
    min_neighbor = min(min_neighbor, v)
    for i in xrange(v):
        # do a length limited BFS:
        a = [(i, 0)]
        p = 0
        vtx_indices[i].add(i)
        prev_ring = set([i])
        cur_ring = 0
        while cur_ring < k or len(vtx_indices[i]) < min_neighbor:
            cur_ring = cur_ring + 1
            new_ring = set([])
            for w in prev_ring:
                for edge, dest in adj[w]:
                    if not (dest in vtx_indices[i]):
                        vtx_indices[i].add(dest)
                        new_ring.add(dest)
            prev_ring = new_ring
                        
                        
        for w in vtx_indices[i]:
            for edge, dest in adj[w]:
                if dest in vtx_indices[i]:
                    edge_indices[i].add(edge)

        anv = anv + len(vtx_indices[i])
        ane = ane + len(edge_indices[i])
        
        vtx_indices[i] = array(list(vtx_indices[i]), 'i')
        edge_indices[i] = array(list(edge_indices[i]), 'i')

    print_info("\taverage #v = %d, average #e = %d" % (anv/v, ane/v))
    return vtx_indices, edge_indices

def estimate_sub_stress_space_from_subgraphs(L_rhos, dim_T, g, vtx_indices, edge_indices):
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
            result = estimate_sub_stress_space(L_rhos, g, edge_indices[i], vtx_indices[i])

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
        stress_samples = svd(stress_samples)[0]

    return stress_samples[:,:SS_SAMPLES]            

            
def sample_from_stress_space(S_basis):
    n_S = S_basis.shape[1]

    print_info("Sampling from stress space...")

    # If you want randomize:
    stress_samples = asmatrix(S_basis) * asmatrix(random.random((n_S, SS_SAMPLES+SS_SAMPLES/2)))

    # Else take the last ss_samples stress vectors in S_basis
    #stress_samples = S_basis[n_S-ss_samples:]

    if ORTHO_SAMPLES:
        stress_samples = svd(stress_samples)[0]

    return stress_samples[:,:SS_SAMPLES]
    

def estimate_stress_kernel(g, stress_samples):
    v, d, E = g.v, g.d, g.E
    ss_samples = stress_samples.shape[1]
    
    n_per_matrix = max(d, min(d*KERNEL_SAMPLES, v/5)) # number of basis vectors to pick from kernel of each matrix
    print_info("Taking %d eigenvectors from each stress matrix" % n_per_matrix)
    stress_kernel = zeros((v, ss_samples * n_per_matrix))

    print_info("Computing kernel for %d stress" % ss_samples)
    sparse_template = sparse_stress_matrix_template(g)

    nzval = zeros(sparse_template[0].shape, 'd')
    eigval = zeros((n_per_matrix+1), 'd')
    eigvec = zeros((v, n_per_matrix+1), 'd')

    for i in xrange(ss_samples):
        w = stress_samples[:, i]

        if USE_SPARSE_EIG:
            # The following uses sparse matrix routines
            stress_matrix_speig_from_vector(w=w,
                                            sparse_template=sparse_template,
                                            nzval = nzval,
                                            nev = n_per_matrix+1,
                                            eigval = eigval,
                                            eigvec = eigvec)
            stress_kernel[:, i*n_per_matrix : (i+1)*n_per_matrix] = eigvec[:,1:]
        else:
            # The following uses dense matrix routines
            omega = stress_matrix_from_vector(w, E, v)
            eigval, eigvec = eig(omega)     # v by v, sparse, 2vd non-zero entries
            eigval = abs(eigval)
        
            order =  range(v)
            order.sort(key = lambda i: eigval[i])

            k = i*n_per_matrix
            stress_kernel[:, k:k+n_per_matrix] = eigvec[:,order[1:n_per_matrix+1]]
            if WEIGHT_KERNEL_SAMPLE:
                for j in xrange(n_per_matrix):
                    stress_kernel[:, k+j] *= -math.log(eigval[order[j + 1]])

        #print eigval[order[0: n_per_matrix]]
        sys.stdout.write('.')
        sys.stdout.flush()

    sys.stdout.write('\n')
    print_info("Calculating dominant stress kernel...")
    stress_kernel_basis, ev, whatever = svd(stress_kernel) # v by C, where C = n_per_matrix * ss_samples, dense

    # Cross match TOP_STRESS_KERENL vectors with the stress samples:
    error = zeros((ss_samples))
    mean_error = []
    for i in xrange(TOP_STRESS_KERNEL):
        for j in xrange(ss_samples):
            m = stress_matrix_from_vector(stress_samples[:,j], E, v)
            error[j] = norm(m * stress_kernel_basis[:,i:i+1])
        mean_error.append(mean(error))
    print_info("Mean norm of product with stress matrix for top agg. kernel:\n\t%s" % str(mean_error))

    return stress_kernel_basis[:, :d], ev

def calculate_relative_positions(g, L_rho, q):
    B = optimal_linear_transform_for_l(q, g.E, L_rho)
    return B * q

def graph_scale(g, perturb, noise_std, sampling, k, min_neighbor, floor_plan):
    v, p, d, e, E = g.v, g.p, g.d, g.e, g.E

    info = 'Graph scale parameters:'
    info += '\n\tv = %d'        % v
    info += '\n\tdimension = %d' % d
    info += '\n\tdist threshold = %1.02f'   % PARAM_DIST_THRESHOLD
    info += '\n\tmax neighbors = %d' % MAX_NEIGHBORS
    info += '\n\tnoise = %.04f (= %.04fm for additive noise)'     % (noise_std, noise_std * METER_RATIO)
    info += '\n\tperturb = %.04f = %.04fm'     % (perturb, perturb * METER_RATIO)
    info += '\n\tsampling = %2.02f'    % sampling
    info += '\n\tkernel samples = %02d'     % KERNEL_SAMPLES
    info += '\n\tortho samples = %s'       % str(ORTHO_SAMPLES)[0]
    info += '\n\tmult noise = %s'       % str(MULT_NOISE)[0]           
    info += '\n\texact local stress = %s'       % str(EXACT_LOCAL_STRESS)[0]   
    info += '\n\tstres space samples = %03d'     % SS_SAMPLES                   
    info += '\n\tweight kernel samples = %s'      % str(WEIGHT_KERNEL_SAMPLE)[0] 
    info += '\n\tstress value perc = %03d'    % STRESS_VAL_PERC              
    info += '\n\ttop stress kernel to pick = %d'      % TOP_STRESS_KERNEL            
    info += '\n\tstress sample method = %s'         % STRESS_SAMPLE
    info += '\n\tfloor plan = %s'                   % FLOOR_PLAN_FN
    print_info(info)


    check_gr(g)

    dim_T = locally_rigid_rank(v, d)

    if STRESS_SAMPLE == 'global':
        n_samples = int(dim_T * sampling)

        L_rhos, L_rho = measure_L_rho(g, perturb, noise_std, n_samples)

        S_basis, cov_spec = estimate_stress_space(L_rhos, dim_T)
        stress_samples = sample_from_stress_space(S_basis)
        
    else:
        vtx_indices, edge_indices = get_k_ring_subgraphs(g, k, min_neighbor)

        n_samples = int(max(map(lambda vi: len(vi) * d, vtx_indices)) * sampling)
        L_rhos, L_rho = measure_L_rho(g, perturb, noise_std, n_samples)

        sub_S_basis, sparse_param = estimate_sub_stress_space_from_subgraphs(
            L_rhos = L_rhos,
            dim_T = dim_T,
            g = g,
            vtx_indices = vtx_indices,
            edge_indices = edge_indices)

        if STRESS_SAMPLE == 'semilocal':
            S_basis, cov_spec = consolidate_sub_space(dim_T, sub_S_basis, sparse_param)
            stress_samples = sample_from_stress_space(S_basis)

        elif STRESS_SAMPLE == 'local':
            cov_spec = zeros((1))
            stress_samples = sample_from_sub_stress_space(sub_S_basis, sparse_param)

    K_basis, stress_spec = estimate_stress_kernel(g, stress_samples)
    
    q = asmatrix(K_basis.T) # coordinate vector

    T_q = calculate_relative_positions(g, L_rho, q)

    l_approx = (optimal_rigid(T_q, p) * homogenous_vectors(T_q))[:d,:]
    af_approx = (optimal_affine(q, p) * homogenous_vectors(q))[:d,:]

    ## Calculate results from trilateration
    ## print_info("Performing trilateration for comparison:")
    ## tri = trilaterate_graph(g.adj, sqrt(L_rho).A.ravel())
    ## tri.p = (optimal_rigid(tri.p[:,tri.localized], p[:,tri.localized]) * homogenous_vectors(tri.p))[:d,:]
    tri = None

    l = L(p, E, 0)

    def mean_l2_error(v1, v2):
        d, n = v1.shape
        return mean(array([norm(v1[:,i] - v2[:,i]) for i in xrange(n)]))
    
    af_g_error = mean_l2_error(p, af_approx)
    af_l_error = mean_l2_error(l.T, L(af_approx, E, 0).T)

    l_g_error = mean_l2_error(p, l_approx)
    l_l_error = mean_l2_error(l.T, L(l_approx, E, 0).T)

    #tri_g_error = mean_l2_error(p[:,tri.localized], tri.p[:,tri.localized])

    #print("\t#localized vertices = %d = (%f%%).\n\tMean error = %f = %fm" %
    #      (len(tri.localized), 100 *float(len(tri.localized))/v, tri_g_error, tri_g_error * METER_RATIO))
    
    # plot the info
    plot_info(g,
              L_opt_p = l_approx,
              aff_opt_p = af_approx,
              tri = tri,
              dim_T = dim_T,
              cov_spec = cov_spec,
              stress_spec = stress_spec,
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

    #floor_plan = FloorPlan(FLOOR_PLAN_FN)
    floor_plan = Floorplan("%s/%s" % (DIR_DATA, FLOOR_PLAN_FN))

    # hand pick spaces to use so can get ggr rooms
    g = random_graph(v = PARAM_V,
                     d = PARAM_D,
                     max_dist = dist_threshold,
                     min_dist = dist_threshold * 0.01,
                     discardNonrigid = True,
                     max_neighbors = MAX_NEIGHBORS,
                     floor_plan = floor_plan)

    edgelen = sqrt(L(g.p, g.E, 0.0).A.ravel())
    edgelen_min, edgelen_med, edgelen_max = min(edgelen), median(edgelen), max(edgelen)
    global METER_RATIO
    METER_RATIO = MAX_EDGE_LEN_IN_METER / edgelen_max
    print_info("Realworld ratio: 1 unit = %fm " % METER_RATIO)
    print_info("Edge length:\n\tmin = %f = %fm\n\tmedian = %f = %fm\n\tmax = %f = %fm" % (
        edgelen_min, edgelen_min * METER_RATIO,
        edgelen_med, edgelen_med * METER_RATIO,
        edgelen_max, edgelen_max * METER_RATIO))
    
    if MULT_NOISE:
        noise_stds = array(PARAM_NOISE_STDS, 'd')
        perturbs = edgelen_med * array(PARAM_PERTURBS, 'd')
    else:
        noise_stds = array(PARAM_NOISE_STDS, 'd') * dist_threshold
        perturbs = array(PARAM_PERTURBS, 'd')

    for noise_std in noise_stds:
        for perturb in perturbs * noise_std:
            for sampling in PARAM_SAMPLINGS:
                e = graph_scale(g = g, 
                                perturb=max(1e-4, perturb),
                                noise_std = noise_std,
                                sampling = sampling,
                                k = K_RING,
                                min_neighbor = MIN_LOCAL_NBHD,
                                floor_plan = floor_plan)
                sys.stdout.flush()


if __name__ == "__main__":
    main()
