#!/usr/bin/python
from numpy import *
from scipy.linalg.basic import *
from scipy.linalg.decomp import *
import scipy.stats
import cPickle
from pylab import *
from matplotlib.collections import LineCollection
from matplotlib.patches import Circle
from geo import *
from util import *
from graph import *
from sparselin import *

EPS = 1e-12


PARAM_V = 400
PARAM_D = 2
PARAM_NOISE_STDS = [0.02]               # [0.02]
PARAM_PERTURBS = [10.0]
PARAM_SAMPLINGS = [2.5]


STRESS_SAMPLE = 'semilocal' # global | semilocal | local
USE_SPARSE_EIG = False
USE_SPARSE_SVD = True

KERNEL_SAMPLES = 32 # Helps a lot going from 2 to 16 for noise cases, was 16
ORTHO_SAMPLES = True # Helps a lot (reduce mean error by 2 times)
MULT_NOISE = False
EXACT_LOCAL_STRESS = False
SS_SAMPLES = 60
WEIGHT_KERNEL_SAMPLE = True

CONSOLIDATE_READ_FROM_CACHE = True
CONSOLIDATE_WRITE_TO_CACHE = True
    

# TODO: wrap in a class, so no g param for all functions

class DOFTooFew(Exception):
    pass

class NotLocallyRigid(Exception):
    pass

class NotGloballyRigid(Exception):
    pass

class TooFewSamples(Exception):
    pass
    

def L(p, E, noise_std):
    """
    p is d x n and encodes n d-dim points. E is k x 2 and encodes the
    edge set. Returns k-dim column vector containing the squared
    distances of the edges.
    """
    v = asarray(p[:, E[:,0]] - p[:, E[:,1]])
    d = sqrt((v * v).sum(axis=0))
    noise = random.randn(len(E)) * noise_std
    if MULT_NOISE:
        d *= 1 + noise
    else:
        d += noise
    return asmatrix(d * d).T

def locally_rigid_rank(v, d):
    return v * d - chooses(d+1,2)

def random_p(v, d):
    return asmatrix(array(random.random((d,v)), order='FORTRAN'))

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

figure(figsize=(12,8))

def plot_info(stats, cov_spec, t, stress_spec, d,
              p, approx_p, approx_p2, v, E):
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
    
    
    #Graph the spectral distribution of the covariance matrix
    axes([margin, margin, width-margin*2, height-margin*2])
    #semilogx()
    plot(xrange(1, 1+len(cov_spec)),cov_spec)
    axvline(E.shape[0]-t+1)
    #axis([1, min(t*16, len(E))+1, 1e-4, 1e1])
    gca().set_aspect('auto')
    title("Cov. Spec.")

    #Graph the spectral distribution of the stress matrix
    axes([margin, height+margin, width-margin*2, height-margin*2])
    #loglog()
    semilogx()
    plot(xrange(1, 1 + len(stress_spec)), stress_spec)
    axvline(d)
    axvline(d+1)
    #axis([1, 1+len(stress_spec), 1e-2, 1e2 ])
    gca().set_aspect('auto')
    title("Agg. Stress Kern. Spec.")

    #Graph the geometry
    axes([margin+width, margin, width*2-margin*2, height*2-margin*2])
    title("Error")
    point_size = 20*math.sqrt(30)/math.sqrt(v)
    scatter(x = approx_p2[0].A.ravel(), y = approx_p2[1].A.ravel(), s = point_size,
            linewidth=(0.0), c = "green", marker = 'o', zorder=99, alpha=0.75)
    scatter(x = approx_p[0].A.ravel(), y = approx_p[1].A.ravel(), s = point_size,
            linewidth=(0.0), c = "r", marker = 'o', zorder=100, alpha=0.75)
    scatter(x = p[0].A.ravel(), y = p[1].A.ravel(), s = point_size,
            linewidth=(0.0), c = "b", marker = 'o', zorder =102, alpha=1)

    
    axis([-0.2, 1.2, -0.2, 1.2])
    gca().set_aspect('equal')

    gca().add_patch(Circle((0.5, 1.1),
                           radius = stats["perturb"],
                           linewidth=0.03 * point_size,
                           fill=False,
                           alpha = 0.5,
                           facecolor = "lightgrey",
                           edgecolor = "black"))

    gca().add_collection(LineCollection([
        p.T[e] for e in E], colors = "lightgrey", alpha=0.75, linewidth=0.03 * point_size))
    gca().add_collection(LineCollection([
        (approx_p2.T[i].A.ravel(), p.T[i].A.ravel())
        for i in xrange(v)], colors = "green", alpha=0.75, linewidth= 0.03 * point_size))
    gca().add_collection(LineCollection([
        (approx_p.T[i].A.ravel(), p.T[i].A.ravel())
        for i in xrange(v)], colors = "red", alpha=0.75, linewidth = 0.03 * point_size))

    fn = 'infoplot-v%d-n%.04f-p%.04f-s%2.02f-ks%02d-os%s-ms%s-el%s-ss%03d-wks%s-%s' % (
        v, stats["noise"], stats["perturb"], stats["sampling"],
        KERNEL_SAMPLES,
        str(ORTHO_SAMPLES)[0],
        str(MULT_NOISE)[0],
        str(EXACT_LOCAL_STRESS)[0],
        SS_SAMPLES,
        str(WEIGHT_KERNEL_SAMPLE)[0],
        STRESS_SAMPLE)
    print_info("%s.eps dumped" % fn)
    import os
    savefig("%s.eps" % fn)
    os.system("epstopdf %s.eps" %fn)
    print_info("%s.pdf generated" % fn)
    os.system("gnome-open %s.pdf" %fn)

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

def estimate_sub_stress_space(L_rhos, g, edge_idx, vtx_idx = None):
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
    return sub_S_basis, s

# This doesn't work for some reason!
def calculate_exact_sub_stress_space(g, edge_idx, vtx_idx = None):
    p, E = g
    if vtx_idx == None:
        vtx_idx = affected_vertices(E, edge_idx)
    v = len(vtx_idx)

    invert_vtx_idx = -ones((v_d_e_from_graph(g)[0]), 'i')
    for i, w in enumerate(vtx_idx):
        invert_vtx_idx[w] = i

    d = v_d_e_from_graph(g)[1]

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
    t = locally_rigid_rank(v, d)
    while t < min(len(edge_idx), v * d):
        if s[t] < EPS:
            break
        t = t + 1

    return u[:,t:], s
    

def get_k_ring_subgraphs(g, k, min_neighbor):
    """
    return k_rings of each vertex. If the #v of the ring is less than
    min_neighbor, additional vertices in larger rings are added till
    at least min_neighbor of them is in the graph. Result is stored
    as two list, one vtx indices, one edge indices.
    """
    print_info("Computing %d-ring subgraphs" % k)
    adj = adjacency_list_from_edge_list(g)
    E = g[1]
    v, d, e = v_d_e_from_graph(g)
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
        while cur_ring < k or len(vtx_indices) < min_neighbor:
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
    v, d, e = v_d_e_from_graph(g)

    print_info("Computing stress space for each subgraph")
    sub_S_basis = []
    n = 0
    nz = 0
    for i in xrange(v):
        if EXACT_LOCAL_STRESS:
            sub_S_basis.append(calculate_exact_sub_stress_space(g, edge_indices[i], vtx_indices[i])[0])
        else:
            sub_S_basis.append(estimate_sub_stress_space(L_rhos, g, edge_indices[i], vtx_indices[i])[0])
        
        n = n + sub_S_basis[i].shape[1]
        nz = nz + sub_S_basis[i].shape[0] * sub_S_basis[i].shape[1]
        sys.stdout.write('.')
        sys.stdout.flush()
    sys.stdout.write('\n')

    return sub_S_basis, (e, n, nz, edge_indices)

def consolidate_sub_space(dim_T, sub_basis, sparse_param):
    e, n, nz, edge_indices = sparse_param

    print_info("Consolidating subspaces...")

    if USE_SPARSE_SVD:
        read_succ = False
        if CONSOLIDATE_READ_FROM_CACHE:
            try:
                fn = "consolidate-pca-%d-%d-%d.cache" % (e, n, nz)
                f = open(fn, "r")
                u, s = cPickle.load(f)
                print_info("\tRead from consolidation PCA cache %s" % fn)
                read_succ = True
            except IOError:
                print_info("\tError reading from consolidation PCA cache %s. Perform PCA..." % fn)

        if (not CONSOLIDATE_READ_FROM_CACHE) or (not read_succ):
            # Now write a temporary file of the big sparse matrix
            print_info("Write out %dx%d sparse matrix for external SVDing" % (e, n) )
            f = open("input.st", "w")
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

            if CONSOLIDATE_WRITE_TO_CACHE:
                fn = "consolidate-pca-%d-%d-%d.cache" % (e, n, nz)
                f = open(fn, "w")
                cPickle.dump((u, s),f)
                f.close()
                print_info("\tWrite to consolidation PCA cache %s" % fn)

        return u, s

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
        return u[:, :e-dim_T], s

    # return m, zeros((e-dim_T))

def sample_from_sub_stress_space(g, sub_S_basis, sparse_param):
    v, d, e = v_d_e_from_graph(g)
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

    p, E = g
    v, d, e = v_d_e_from_graph(g)
    ss_samples = stress_samples.shape[1]
    
    n_per_matrix = d*KERNEL_SAMPLES # number of basis vectors to pick from kernel of each matrix
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
    return stress_kernel_basis[:, :d], ev

def calculate_relative_positions(g, L_rho, q):
    B = optimal_linear_transform_for_l(q, g[1], L_rho)
    return B * q

def graph_scale(g, perturb, noise_std, sampling, k, min_neighbor):

    print_info('Graph Scale PARAMS:\n\tv=%d\n\tnoise=%.04f\n\tperturb=%.04f\n\tsampling=%2.02f\n\tkernel_samples=%2d\n\tortho_samples=%s\n\tmult_noise=%s\n\texact_local_stress=%s\n\tstress_samples=%3d\n\tstress_sample_method=%s\n\tweight_kernel_sample=%s' % (
        v_d_e_from_graph(g)[0], noise_std, perturb, sampling,
        KERNEL_SAMPLES,
        str(ORTHO_SAMPLES)[0],
        str(MULT_NOISE)[0],
        str(EXACT_LOCAL_STRESS)[0],
        SS_SAMPLES,
        STRESS_SAMPLE,
        str(WEIGHT_KERNEL_SAMPLE)[0]))

    p, E = g
    v, d, e = v_d_e_from_graph(g)

    check_ggr(g)

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

        sub_S_basis, sparse_param = estimate_sub_stress_space_from_subgraphs(L_rhos, dim_T, g, vtx_indices, edge_indices)

        if STRESS_SAMPLE == 'semilocal':
            S_basis, cov_spec = consolidate_sub_space(dim_T, sub_S_basis, sparse_param)
            stress_samples = sample_from_stress_space(S_basis)

        elif STRESS_SAMPLE == 'local':
            cov_spec = zeros((1))
            stress_samples = sample_from_sub_stress_space(g, sub_S_basis, sparse_param)

    K_basis, stress_spec = estimate_stress_kernel(g, stress_samples)
    
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
               "samples":n_samples,
               "l g error": l_g_error,
               "l l error": l_l_error,
               "af g error": af_g_error,
               "af l error": af_l_error},
              cov_spec = cov_spec, t = dim_T,
              stress_spec = stress_spec, d = d,
              p = p, approx_p = l_approx, approx_p2 = af_approx, v = v, E = E)

    print_info("Mean error %f" % l_g_error)

    return l_g_error

def main():
    import sys
    random.seed(0)
    k =  math.pow(2*(PARAM_D+1), 1.0/PARAM_D)/3.0
    dist_threshold = 4*k*math.pow(PARAM_V, -1.0/PARAM_D)

    g = random_graph(v = PARAM_V,
                     d = PARAM_D,
                     max_dist = dist_threshold,
                     min_dist = dist_threshold * 0.01,
                     discardNonrigid = True,
                     max_neighbors = 10)

    noise_stds = array(PARAM_NOISE_STDS, 'd') * dist_threshold * 0.5
    perturbs = array(PARAM_PERTURBS, 'd') 
    if MULT_NOISE:
        noise_stds /= (dist_threshold * 0.5)
        perturbs *= (0.5  * dist_threshold)
        
    for noise_std in noise_stds:
        for perturb in perturbs * noise_std:
            for sampling in PARAM_SAMPLINGS:
                e = graph_scale(g = g, 
                                perturb=max(1e-4, perturb),
                                noise_std = noise_std,
                                sampling = sampling,
                                k = 2,
                                min_neighbor = 30)
                sys.stdout.flush()


if __name__ == "__main__":
    main()
