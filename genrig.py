from settings import *
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
        d *= (1 + noise)
    else:
        d += noise
    return asmatrix(d * d).T

class GenericRigidity:

    def __init__(self, v, d, E, eps = EPS, rigidity_iter = 3, stress_iter = 3):
        print_info('Calculating generic rigidity...')
        t = locally_rigid_rank(v, d)
        e = len(E)
        rigidity_rank = 0
        stress_kernel_dim = e

        for x in xrange(rigidity_iter):
            D = rigidity_matrix(v, d, E)
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

        self.rigidity_matrix_rank = rigidity_rank
        self.stress_kernel_dim = stress_kernel_dim
        
        if rigidity_rank != t:
            self.type = 'N'
        elif stress_kernel_dim != d + 1:
            self.type = 'L'
        else:
            self.type = 'G'

        print_info('\ttype = %s\n\trigidity matrix rank = %d  (max = %d)\n\tstress kernel dim = %d (min = %d)'
                   % (self.type, rigidity_rank, t, stress_kernel_dim, d + 1))

