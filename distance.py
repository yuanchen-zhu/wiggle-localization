#!/usr/bin/python
from numpy import *
from scipy.linalg.basic import *
from scipy.linalg.decomp import *
import pickle, scipy.stats
from pylab import *
from matplotlib.collections import LineCollection
from geo import *
from util import *
from graph import *
from sparselin import *

EPS = 1e-12

def L(p, E, noise_std):
    """
    p is d x n and encodes n d-dim points. E is k x 2 and encodes the
    edge set. Returns k-dim column vector containing the squared
    distances of the edges.
    """
    v = asarray(p[:, E[:,0]] - p[:, E[:,1]])
    d = sqrt((v * v).sum(axis=0))
    noise = random.randn(len(E)) * noise_std
    d += noise
    return asmatrix(d * d).T

def locally_rigid_rank(v, d):
    return v * d - chooses(d+1,2)

def random_p(v, d):
    return asmatrix(array(random.random((d,v)), order='FORTRAN'))

class DOFTooFew(Exception):
    pass

class NotLocallyRigid(Exception):
    pass

class NotGloballyRigid(Exception):
    pass


def random_graph(v, d, max_dist, min_dist, discardNonrigid = True, max_neighbors = 12):
    i = 0
    while 1:
        p = random_p(v, d)
        E = build_edgeset(p, max_dist, min_dist, max_neighbors)
        e = len(E)
        i = i + 1
        if discardNonrigid:
            t = locally_rigid_rank(v, d)
            if t >= e:
                raise DOFTooFew()
            rig =  rigidity(v, d, E)
            if rig != "G":
                continue
        if discardNonrigid:
            print_info("Globally rigid graph created after %d tries" % i)

        print_info("|V|=%d |E|=%d" % (v, e))
        return p, E


def rigidity(v, d, E, eps = EPS, rigidity_iter = 2, stress_iter = 2):
    print_info("Calculating rigidity...")
    t = locally_rigid_rank(v, d)
    e = len(E)
    rigidity_rank = 0
    stress_kernel_dim = e

    for x in xrange(rigidity_iter):
        p = random_p(v, d)
        D = zeros((len(E), d * v))
        for k in xrange(len(E)):
            i, j = E[k][0], E[k][1]
            diff_ij = p[:,i] - p[:,j]
            D[k, i*d:i*d+d] = 2.0 * diff_ij.T
            D[k, j*d:j*d+d] = -2.0 * diff_ij.T
        u, s, vh = svd(D)               # E by dv, sparse, 2dE non-zero
        rigidity_rank = max(rigidity_rank, len(s[s >= eps]))

        if rigidity_rank != t:
            continue

        stress_basis = u[:,t:]
        for y in xrange(stress_iter):
            w = stress_basis * asmatrix(random.random((e - t, 1)))
            w /= norm(w)
            omega = stress_matrix_from_vector(w, E, v)
            eigval, eigvec = eig(omega) # sparse
            abs_eigval = abs(eigval)
            stress_kernel_dim = min(stress_kernel_dim,
                                    len(abs_eigval[abs_eigval <= eps]))

        if rigidity_rank == t and stress_kernel_dim == d + 1:
            break
    
    if rigidity_rank != t:
        return "N"
    elif stress_kernel_dim != d + 1:
        return "L"
    else:
        return "G"

def stress_matrix_from_vector(w, E, v):
    O = asmatrix(zeros((v,v), 'd'))
    V = xrange(v)
    for i in xrange(len(E)):
        p, q = E[i][0], E[i][1]
        O[p,q] = O[q,p] = w[i]
        O[p,p] -= w[i]
        O[q,q] -= w[i]
    return O

def sparse_stress_matrix_template(g):
    adj = adjacency_list_from_edge_list(g)
    v, d, e = v_d_e_from_graph(g)

    nzval_idx = []
    irow = []
    pcol = [0]
    for i in xrange(v):
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
        

def stress_matrix_eig_from_vector(w, sparse_template, nzval, nev, eigval, eigvec):
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
          maxit=len(pcol)*100)

    order =  range(nev)
    order.sort(key = lambda i: abs(eigval[i]))
    eigval[:] = eigval[order]
    eigvec[:,:] = eigvec[:,order]

    #print eigval

figure(figsize=(12,8))

def plot_info(stats, cov_spec, t, stress_spec, d,
              p, approx_p, approx_p2, v, E):
    margin = 0.05
    width = 1.0/3.0
    height = (1.0-margin*2)/2.0

    clf()

    axes([0,0,1,1-margin*2], frameon=False, xticks=[], yticks=[])
    title('Noise:%.04f   Kern.:%.04f   Sampling:%2.02f'
          % (stats["noise"], stats["perturb"], stats["sampling"]))
    figtext(width*2-margin, height*2-margin*1.5,
            'aff. err: %.06f\naff. L-err: %.06f'
            % (stats["af g error"], stats["af l error"]),
            ha = "right", va = "top", color = "green")
    figtext(width*3-margin*2, height*2-margin*1.5,
            'err: %.06f\nL-err: %.06f' %
            (stats["l g error"], stats["l l error"]),
            ha = "right", va = "top", color = "red")
    
    
    #Graph the spectral distribution of the covariance matrix
    axes([margin, margin, width-margin*2, height-margin*2])
    loglog()
    plot(xrange(1, 1+len(cov_spec)),cov_spec)
    axvline(t+1)
    axis([1, min(t*16, len(E))+1, 1e-4, 1e1])
    gca().set_aspect('auto')
    title("Cov. Spec.")

    #Graph the spectral distribution of the stress matrix
    axes([margin, height+margin, width-margin*2, height-margin*2])
    loglog()
    plot(xrange(1, 1 + len(stress_spec)), stress_spec)
    axvline(d)
    axvline(d+1)
    axis([1, 1+len(stress_spec), 1e-2, 1e2 ])
    gca().set_aspect('auto')
    title("Agg. Stress Kern. Spec.")

    #Graph the geometry
    axes([margin+width, margin, width*2-margin*2, height*2-margin*2])
    title("Error")
    scatter(x = approx_p2[0].A.ravel(), y = approx_p2[1].A.ravel(), s = 16,
            linewidth=(0.0), c = "green", marker = 'o', zorder=99, alpha=0.75)
    scatter(x = approx_p[0].A.ravel(), y = approx_p[1].A.ravel(), s = 16,
            linewidth=(0.0), c = "r", marker = 'o', zorder=100, alpha=0.75)
    scatter(x = p[0].A.ravel(), y = p[1].A.ravel(), s = 16,
            linewidth=(0.0), c = "b", marker = 'o', zorder =102, alpha=1)
    axis([-0.2, 1.2, -0.2, 1.2])
    gca().set_aspect('equal')

    gca().add_collection(LineCollection([
        p.T[e] for e in E], colors = "lightgrey", alpha=0.75))
    gca().add_collection(LineCollection([
        (approx_p2.T[i].A.ravel(), p.T[i].A.ravel())
        for i in xrange(v)], colors = "green", alpha=0.75))
    gca().add_collection(LineCollection([
        (approx_p.T[i].A.ravel(), p.T[i].A.ravel())
        for i in xrange(v)], colors = "red", alpha=0.75))

    fn = 'infoplot-v%d-n%.04f-k%.04f-s%2.02f.eps' % (
        v, stats["noise"], stats["perturb"], stats["sampling"])
    print_info("%s dumped" % fn)
    savefig(fn)

def check_ggr(g):
    p, E = g
    v, d, e = v_d_e_from_graph(g)
    
    # Sanity check: is the graph generic globally rigid
    dim_T = locally_rigid_rank(v, d)
    if dim_T >= e:
        raise DOFTooFew()
    else:
        rig =  rigidity(v, d, E)
        if rig == "N": raise NotLocallyRigid()
        elif rig == "L": raise NotGloballyRigid()

def measure_L_rho(g, perturb, noise_std, n_samples):
    p, E = g
    v, d, e = v_d_e_from_graph(g)

    print_info("#measurements = %d" % n_samples)

    L_rhos = asmatrix(zeros((e, n_samples), 'd'))
    for i in xrange(n_samples):
        delta = asmatrix(random.random((d,v))) - 0.5
        delta *= (perturb*2)
        L_rhos[:,i] = L(p + delta, E, noise_std)

    return L_rhos, L(p, E, noise_std)

def estimate_stress_space(L_rhos, dim_T):
    u, s, vh = svd(L_rhos)              # dense
    S_basis = asmatrix(u[:,dim_T:])
    return S_basis, s

def affected_vertices(E, edge_idx):
    return set(E[edge_idx,:].ravel())

def estimate_sub_stress_space(L_rhos, dim_T, g, edge_idx, vtx_idx = None):
    """
    edge_idx is a 1-d array of edge indices. Only the indexed edges
    have the correponding components of the stress space calculated.
    """
    u, s, vh = svd(L_rhos[edge_idx, :]) # dense

    if vtx_idx != None:
        v = len(vtx_idx)
    else:
        v = len(affected_vertices(g[1], edge_idx))
    d = v_d_e_from_graph(g)[1]

    # sanity check for subgraph
    sub_dim_T = locally_rigid_rank(v, d)
    if sub_dim_T >= len(edge_idx) or sub_dim_T >= L_rhos.shape[1]:
        return zeros((len(edge_idx), 0)), zeros((0))

    # Now find the actual dimension of the tangent space, it should be
    # in the range of [locally_rigid_rank(v, d), v * d]
    prev_derivative = 1                   # derivative should always < 0
    for t in xrange(sub_dim_T, min(len(edge_idx), L_rhos.shape[1], v * d)):
        derivative = math.log(s[t] / s[t-1]) / math.log(float(t)/float(t-1))
        if derivative > prev_derivative:
            break
        prev_derivative = derivative

    sub_S_basis = asmatrix(u[:,t-1:])
    return sub_S_basis, s

def get_k_ring_subgraphs(g, k):
    """
    return k_rings of each vertex. Result is stored as two list, one
    vtx indices, one edge indices.
    """
    print_info("Computing %d-ring subgraphs" % k)
    adj = adjacency_list_from_edge_list(g)
    E = g[1]
    v, d, e = v_d_e_from_graph(g)
    vtx_indices = [set([]) for i in xrange(v)]
    edge_indices = [set([]) for i in xrange(v)]
    anv = 0
    ane = 0
    for i in xrange(v):
        # do a length limited BFS:
        a = [(i, 0)]
        p = 0
        vtx_indices[i].add(i)
        while p != len(a):
            w, l = a[p]
            p = p + 1
            if l < k:
                for edge, dest in adj[w]:
                    if not (dest in vtx_indices[i]):
                        a.append((dest, l+1))
                        vtx_indices[i].add(dest)
                        
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


def estimate_stress_space_from_k_ring_subgraphs(L_rhos, dim_T, g, k):
    v, d, e = v_d_e_from_graph(g)
    vtx_indices, edge_indices = get_k_ring_subgraphs(g, k)

    print_info("Computing stress space for each subgraph")
    sub_S_basis = []
    n = 0
    nz = 0
    for i in xrange(v):
        sub_S_basis.append(estimate_sub_stress_space(L_rhos, dim_T, g, edge_indices[i], vtx_indices[i])[0])
        n = n + sub_S_basis[i].shape[1]
        nz = nz + sub_S_basis[i].shape[0] * sub_S_basis[i].shape[1]

    # Now write a temporary file of the big sparse matrix
    print_info("Write out %dx%d sparse matrix for external SVDing" % (e, n) )
    f = open("input.st", "w")
    f.write("%d %d %d\n" % (e, n, nz))
    for k, B in enumerate(sub_S_basis):
        for i in xrange(B.shape[1]):
            f.write("%d" % B.shape[0])
            for j in xrange(B.shape[0]):
                f.write(" %d %f" % (edge_indices[k][j], B[j,i]))
            f.write("\n")
    f.close()

    # and use './svd' to svd factorize it
    import os
    os.system("./svd input.st -o output -d %d " % (e-dim_T))

    print_info("Read back SVD result")
    f = open("output-Ut", "r")      # read in columns
    f.readline()                        # skip first line (dimension info)
    u = zeros((e, e-dim_T), "d")
    for i in xrange(e-dim_T):           # now grab each column
        toks = string.split(f.readline())
        c = array([map(float, toks)]).T
        u[:,i] = c[:,0]
    f.close()

    f = open("output-S", "r")           # read in singular values
    f.readline()
    s = array(map(float, string.split(f.read())))

    return u, s
    
    ## @@@ The following uses scipy dense matrix routines

    ## # Now unpack the various sub_S_basis into one big matrix
    ## m = zeros((e, n))
    ## j = 0;
    ## for i in xrange(v):
    ##     j2 = j + sub_S_basis[i].shape[1]
    ##     m[edge_indices[i], j:j2] = sub_S_basis[i]
    ##     j = j2

    ## # finally return qr decomp on the big matrix
    ## print m.shape

    ##u, s, vh = svd(m) # e by Cv, sparse, C^2 v non zero entries, C around 50
    #return u[:, :e-dim_T], s
    
def estimate_stress_kernel(g, S_basis):

    p, E = g
    v, d, e = v_d_e_from_graph(g)
    
    n_S = S_basis.shape[1]

    ss_samples = min(n_S, 300)

    # If you want randomize:
    stress_space = svd(asmatrix(S_basis) *
                       asmatrix(random.random((n_S, ss_samples+ss_samples/2))))[0][:,:ss_samples] # dense
    # Else take the last ss_samples stress vectors in S_basis
    #stress_space = S_basis[n_S-ss_samples:]

    n_per_matrix = d*4 # number of basis vectors to pick from kernel of each matrix
    stress_kernel = zeros((v, ss_samples * n_per_matrix))

    stress_mul = ones(ss_samples)
    # for i in xrange(stress_space.shape[1]):
    #    j = t + i
    #    if j >= len(s):
    #        j = len(s) - 1
    #    stress_mul[i] = -math.log(s[j])

    print_info("stress kernel samples = %d" % ss_samples)
    sparse_template = sparse_stress_matrix_template(g)

    nzval = zeros(sparse_template[0].shape, 'd')
    eigval = zeros((n_per_matrix+1), 'd')
    eigvec = zeros((v, n_per_matrix+1), 'd')
    
    for i in xrange(ss_samples):
        w = stress_space[:, i]

        # The following uses sparse matrix routines
        ## stress_matrix_eig_from_vector(w=w,
        ##                               sparse_template=sparse_template,
        ##                               nzval = nzval,
        ##                               nev = n_per_matrix+1,
        ##                               eigval = eigval,
        ##                               eigvec = eigvec)
        ## stress_kernel[:, i*n_per_matrix : (i+1)*n_per_matrix] = eigvec[:,1:] * stress_mul[i]

        # The following uses dense matrix routines
        omega = stress_matrix_from_vector(w, E, v)
        eigval, eigvec = eig(omega)     # v by v, sparse, 2vd non-zero entries
        eigval = abs(eigval)
    
        order =  range(v)
        order.sort(key = lambda i: eigval[i])
        stress_kernel[:, i*n_per_matrix : (i+1)*n_per_matrix] = eigvec[:,order[1:n_per_matrix+1]] * stress_mul[i]

        sys.stdout.write('.')
        sys.stdout.flush()


    print_info("Calculating dominant stress kernel...")
    stress_kernel_basis, ev, whatever = svd(stress_kernel) # v by C, where C = n_per_matrix * ss_samples, dense
    return stress_kernel_basis[:, :d], ev

def calculate_relative_positions(g, L_rho, q):
    B = optimal_linear_transform_for_l(q, g[1], L_rho)
    return B * q

def graph_scale(g, perturb, noise_std, sampling, k):
    p, E = g
    v, d, e = v_d_e_from_graph(g)

    check_ggr(g)

    dim_T = locally_rigid_rank(v, d)
    n_samples = int(dim_T * sampling)

    L_rhos, L_rho = measure_L_rho(g, perturb, noise_std, n_samples)
    #S_basis, cov_spec = estimate_stress_space(L_rhos, dim_T)
    #S_basis, cov_spec = estimate_sub_stress_space(L_rhos, dim_T, g, range(len(E)))
    S_basis, cov_spec = estimate_stress_space_from_k_ring_subgraphs(L_rhos, dim_T, g, k)

    K_basis, stress_spec = estimate_stress_kernel(g, S_basis)
    
    q = asmatrix(K_basis.T) # coordinate vector

    T_q = calculate_relative_positions(g, L_rho, q)

    l_approx = (optimal_rigid(T_q, p) * homogenous_vectors(T_q))[:d,:]
    af_approx = (optimal_affine(q, p) * homogenous_vectors(q))[:d,:]


    l = L(p, E, 0)

    def mean_l2_error(v1, v2):
        d, n = v1.shape
        return mean(array([norm(v1[:,i] - v2[:,i]) for i in xrange(n)]))
    
    af_g_error = mean_l2_error(p, af_approx)
    af_l_error = mean_l2_error(l.T, L(af_approx, E, 0).T)

    l_g_error = mean_l2_error(p, l_approx)
    l_l_error = mean_l2_error(l.T, L(l_approx, E, 0).T)
    
    # plot the info
    plot_info({"noise":noise_std,
               "perturb":perturb,
               "sampling":sampling,
               "l g error": l_g_error,
               "l l error": l_l_error,
               "af g error": af_g_error,
               "af l error": af_l_error},
              cov_spec = cov_spec, t = dim_T,
              stress_spec = stress_spec, d = d,
              p = p, approx_p = l_approx, approx_p2 = af_approx, v = v, E = E)

    return af_g_error

def main():
    import sys
    random.seed(0)
    v = 400
    d = 2
    k =  math.pow(2*(d+1), 1.0/d)/3.0
    dist_threshold = 4*k*math.pow(v, -1.0/d)
    n_tests = 1
    
    print "#V = %d  D = %d  max_dist = %g  n_tests = %d " % (
        v, d, dist_threshold, n_tests)

    g = random_graph(v, d, dist_threshold, 0.01, discardNonrigid = True, max_neighbors = 10)

    ## noise_stds = array([0.01, 0.02])
    ## perturbs = array([5, 10])
    ## samplings = array([1.5, 2, 4, 8])

    noise_stds = array([0.01])
    perturbs = array([10])
    samplings = array([0.5])
    
    for noise_std in  noise_stds * dist_threshold:
        for perturb in perturbs * noise_std:
            for sampling in samplings:
                e = graph_scale(g = g, 
                                perturb=max(1e-4, perturb),
                                noise_std = noise_std,
                                sampling = sampling,
                                k = 3)
                sys.stdout.flush()


if __name__ == "__main__":
    main()
