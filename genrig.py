import settings as S
from numpy import *
from util import *
import stress
import substress

def locally_rigid_rank(v, d):
    return v * d - chooses(d+1,2)

def rigidity_matrix(v, d, E, p = None):
    if p == None:
        p = random_p(v, d, None)
    D = zeros((len(E), d * v))
    for k in xrange(len(E)):
        i, j = E[k][0], E[k][1]
        diff_ij = p[:,i] - p[:,j]
        D[k, i*d:i*d+d] = 2.0 * diff_ij.T
        D[k, j*d:j*d+d] = -2.0 * diff_ij.T
    return D

def L_map(p, E, noise_std, mult_noise):
    """
    p is d x n and encodes n d-dim points. E is k x 2 and encodes the
    edge set. Returns k-dim column vector containing the squared
    distances of the edges.
    """
    v = asarray(p[:, E[:,0]] - p[:, E[:,1]])
    d = sqrt((v * v).sum(axis=0))
    if noise_std > S.EPS:
        noise = random.normal(loc = 0.0, scale=noise_std, size=(len(E)))
    else:
        noise = zeros((len(E)))
    if mult_noise:
        d *= (1 + noise)
    else:
        d += noise
    return asmatrix(d * d).T

class GenericRigidity:

    def __init__(self, v, d, E, eps = S.EPS, rigidity_iter = 1, stress_iter =1):
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
                w = stress_basis * asmatrix(random.uniform(size=(e - rigidity_rank, 1)))

                if norm(w) > eps:
                    w /= norm(w)
                
                omega = stress.matrix_from_vector(w, E, v)
                kern, oev = stress.Kernel(omega).extract_by_threshold(d)
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


class GenericSubstressRigidity:
    def __init__(self, g, Vs, Es, params, eps = S.EPS):
        import sys
        print_info('Calculating generic rigidity using substresses...')

        v, e = g.v, g.e

        p = random_p(g.v, g.d, None)

        print_info("\tComputing stress space for each subgraph")
        sub_S_basis = []
        n = 0
        nz = 0
        for i in xrange(v):
            try:
                B, s, misdim = substress.calculate_exact_space(g, Es[i], Vs[i], p)
            except substress.TooFewSamples:
                B = zeros((0,0))

            sub_S_basis.append(B)

            
            nz += B.shape[0] * B.shape[1]
            n += B.shape[1]
            sys.stdout.write('.')
            sys.stdout.flush()

        sys.stdout.write('\n')
        sparse_param = (e, n, nz, Es)

        S_basis, stress_var = substress.consolidate(g.gr.dim_T, sub_S_basis, sparse_param, params)
        ss = stress.sample(S_basis, params)


        kern = stress.sample_kernel(g, ss)
        K_basis, stress_spec = kern.extract_by_threshold(g.d, eps=1e-5)

        self.K_basis = K_basis
        self.dim_K = K_basis.shape[1]
        print_info('\tstress kernel dim = %d (min = %d)' % (self.dim_K, g.d))

