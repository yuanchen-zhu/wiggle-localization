import cPickle
import settings as S
from geo import *
from util import *
from graph import *
from tri import *
from plot import *
from genrig import *
import stress
from numpy import *
import string, math

from scipy.linalg.basic import *
from scipy.linalg.decomp import *

class TooFewSamples(Exception):
    pass
   
def affected_vertices(E, edge_idx):
    return set(E[edge_idx,:].ravel())

def estimate_space(Ls, g, edge_idx, vtx_idx = None):
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


def calculate_exact_space(g, edge_idx, vtx_idx = None, replaceP = None):
    E = g.E
    if replaceP == None:
        p = g.p
    else:
        p = replaceP
    if vtx_idx == None:
        vtx_idx = affected_vertices(E, edge_idx)
    v = len(vtx_idx)

    r = ridx(vtx_idx, g.v)

    d = g.d
    newE = E[edge_idx]

    D = rigidity_matrix(v, d, [[r[e[0]], r[e[1]]] for e in newE], g.p[:,vtx_idx])
    u, s, vh = svd_conv(D)               # E by dv, sparse, 2dE non-zero
    t = len(s[s >= S.EPS * 10])
    return u[:,t:], s, (locally_rigid_rank(v, d)-t)

def estimate_space_from_subgraphs(Ls, g, Vs, Es):
    v, e = g.v, g.e

    print_info("Computing stress space for each subgraph")
    sub_S_basis = []
    n = 0
    nz = 0
    missing_stress = []
    for i in xrange(v):
        if S.EXACT_STRESS:
            basis, s, misdim = calculate_exact_space(g, Es[i], Vs[i])
        else:
            basis, s, misdim = estimate_space(Ls, g, Es[i], Vs[i])

        sub_S_basis.append(basis)
        missing_stress.append(misdim)
        
        nz += sub_S_basis[i].shape[0] * sub_S_basis[i].shape[1]
        n += sub_S_basis[i].shape[1]
        sys.stdout.write('.')
        sys.stdout.flush()
    sys.stdout.write('\n')

    missing_stress = array(missing_stress, 'd')
    print_info("Local stress space lost dim:\n\tavg = %f\n\tmax = %f" % (
        mean(missing_stress), max(missing_stress)))

    return sub_S_basis, (e, n, nz, Es)

PCA_CUTOFF = 0 # global stats reported by consolidate

def consolidate(dim_T, sub_basis, sparse_param):
    e, n, nz, Es = sparse_param

    print_info("Consolidating subspaces...")

    if S.USE_SPARSE_SVD:
        read_succ = False
        fn = "%s/consolidate-%s.cache" % (S.DIR_CACHE, get_settings_hash())
        if S.CONSOLIDATE_READ_FROM_CACHE:
            try:
                f = open(fn, "r")
                u, s = cPickle.load(f)
                print_info("\tRead from consolidation S.PCA cache %s" % fn)
                read_succ = True
            except IOError:
                print_info("\tError reading from consolidation PCA cache %s. Perform PCA..." % fn)

        if (not S.CONSOLIDATE_READ_FROM_CACHE) or (not read_succ):
            # Now write a temporary file of the big sparse matrix
            print_info("Write out %dx%d sparse matrix for external SVDing" % (e, n) )
            f = open("%s/input.st" % S.DIR_TMP, "w")
            f.write("%d %d %d\n" % (e, n, nz))
            for k, B in enumerate(sub_basis):
                for i in xrange(B.shape[1]):
                    f.write("%d" % B.shape[0])
                    for j in xrange(B.shape[0]):
                        f.write(" %d %f" % (Es[k][j], B[j,i]))
                    f.write("\n")
            f.close()

            # and use './svd' to svd factorize it
            import os
            os.system("%s/svd %s/input.st -o %s/output -d %d " % (S.DIR_BIN, S.DIR_TMP, S.DIR_TMP, e-dim_T))

            print_info("Read back SVD result")
            f = open("%s/output-Ut" % S.DIR_TMP, "r")      # read in columns
            f.readline()                        # skip first line (dimension info)
            u = zeros((e, e-dim_T), "d")
            for i in xrange(e-dim_T):           # now grab each column
                toks = string.split(f.readline())
                c = array([map(float, toks)]).T
                u[:,i] = c[:,0]
            f.close()

            f = open("%s/output-S" % S.DIR_TMP, "r")           # read in singular values
            f.readline()
            s = array(map(float, string.split(f.read())))
            f.close()

            if S.CONSOLIDATE_WRITE_TO_CACHE:
                f = open(fn, "w")
                cPickle.dump((u, s),f)
                f.close()
                print_info("\tWrite to consolidation PCA cache %s" % fn)
    else:
    
        #The following uses scipy dense matrix routines

        # Now unpack the various sub_basis into one big matrix
        m = zeros((e, n))
        j = 0;
        for i in xrange(len(sub_basis)):
            j2 = j + sub_basis[i].shape[1]
            m[Es[i], j:j2] = sub_basis[i]
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

    thr = median(s[:e-dim_T]) * S.STRESS_VAL_PERC /100
    i = len(s) - 1
    while  i >= 0 and s[i] < thr:
        i = i-1
    print_info("Top %d (%f%%) of %d stresses used" % (i+1, 100*float(i+1)/(e-dim_T), e-dim_T))
    global PCA_CUTOFF
    PCA_CUTOFF = i+1
    
    return u[:,:i+1], s

def sample(sub_S_basis, sparse_param):
    e, n, nz, Es = sparse_param

    print_info("Sampling from sub stress spaces...")
    stress_samples = zeros((e, S.SS_SAMPLES+S.SS_SAMPLES/2), order = 'FORTRAN')
    for i in xrange(S.SS_SAMPLES+S.SS_SAMPLES/2):
        for j, basis in enumerate(sub_S_basis):
            w = asmatrix(basis) * asmatrix(random.random((basis.shape[1], 1)))
            w /= norm(w)
            stress_samples[Es[j],i:i+1] += w

    if S.ORTHO_SAMPLES:
        stress_samples = svd(stress_samples)[0][:,:stress_samples.shape[1]]

    return stress_samples[:,:S.SS_SAMPLES]            


