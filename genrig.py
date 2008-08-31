import settings as S
from numpy import *
from util import *
from stress import *

def locally_rigid_rank(v, d):
    return v * d - chooses(d+1,2)

def rigidity_matrix(v, d, E):
    p = random_p(v, d, None)
    D = zeros((len(E), d * v))
    for k in xrange(len(E)):
        i, j = E[k][0], E[k][1]
        diff_ij = p[:,i] - p[:,j]
        D[k, i*d:i*d+d] = 2.0 * diff_ij.T
        D[k, j*d:j*d+d] = -2.0 * diff_ij.T
    return D

def L_map(p, E, noise_std):
    """
    p is d x n and encodes n d-dim points. E is k x 2 and encodes the
    edge set. Returns k-dim column vector containing the squared
    distances of the edges.
    """
    v = asarray(p[:, E[:,0]] - p[:, E[:,1]])
    d = sqrt((v * v).sum(axis=0))
    noise = random.randn(len(E)) * noise_std
    if S.MULT_NOISE:
        d *= (1 + noise)
    else:
        d += noise
    return asmatrix(d * d).T

class GenericRigidity:

    def __init__(self, v, d, E, eps = S.EPS, rigidity_iter = 3, stress_iter = 3):
        print_info('Calculating generic rigidity...')
        if v*(v-1)/2 == d:
            self.type = 'G'
            return
        
        t = locally_rigid_rank(v, d)
        e = len(E)
        rigidity_rank = 0
        dim_K = v + 1

        for x in xrange(rigidity_iter):
            D = rigidity_matrix(v, d, E)
            u, s, vh = svd(D)               # E by dv, sparse, 2dE non-zero
            rigidity_rank = max(rigidity_rank, len(s[s >= eps]))

            stress_basis = u[:,rigidity_rank:]
            for y in xrange(stress_iter):
                w = stress_basis * asmatrix(random.random((e - rigidity_rank, 1)))
                
                w /= norm(w)
                
                omega = stress_matrix_from_vector(w, E, v)
                kern, oev = calculate_single_stress_kernel(omega)
                if dim_K > kern.shape[1]+1:
                    dim_K = kern.shape[1]+1
                    self.K_basis = kern

                if rigidity_rank == t and dim_K == d + 1:
                    break

        self.dim_T = rigidity_rank
        self.dim_K = dim_K
        
        if rigidity_rank != t:
            self.type = 'N'
        elif dim_K != d + 1:
            self.type = 'L'
        else:
            self.type = 'G'

        print_info('\ttype = %s\n\trigidity matrix rank = %d  (max = %d)\n\tstress kernel dim = %d (min = %d)'
                   % (self.type, rigidity_rank, t, dim_K, d + 1))
