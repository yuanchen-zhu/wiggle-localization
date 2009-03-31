from numpy import *
import settings as S
from util import *
from scipy.linalg.basic import *
import sys
import genrig
from symeig import symeig

def estimate_space(Ls, dim_T):
    u, s, vh = svd(Ls)              # dense
    return u[:,dim_T:], s

def calculate_exact_space(g):
    D = genrig.rigidity_matrix(g.v, g.d, g.E, g.p)
    u, s, vh = svd_conv(D)               # E by dv, sparse, 2dE non-zero
    t = len(s[s >= S.EPS])
    return u[:,t:], s

            
def sample(S_basis):
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

def matrix_from_vector(w, E, v):
    O = asmatrix(zeros((v,v), 'd'))
    V = xrange(v)
    for i in xrange(len(E)):
        p, q = E[i][0], E[i][1]
        O[p,q] = O[q,p] = w[i]
        O[p,p] -= w[i]
        O[q,q] -= w[i]
    return O

def is_nan(f):
    return type(f) == 'float' and f != f

class NoValidEigenValueFound:
    pass

class Kernel:
    def __init__(self, sigma):
        self.v = sigma.shape[0]
        self.sigma = asmatrix(sigma)
        self.eigval, self.eigvec = eigh(self.sigma)

        self.eigval = abs(self.eigval)
        order = range(self.v)
        order.sort(key = lambda i : self.eigval[i])
        self.eigval = self.eigval[order]
        self.eigvec = self.eigvec[:, order]
        
    def extract(self, eps = S.EPS):
        kd = len(self.eigval[self.eigval < eps])
        return self.eigvec[:,1:kd], self.eigval[1:]

    def extract_sub(self, sub_dims, kern_dim_minus_one, eps = S.EPS):
        if len(sub_dims) == 1:
            return zeros((1, kern_dim_minus_one), 'd'), zeros((1,1), 'd')

        basis = asmatrix(ones(((self.v), 1), 'd')* (1.0/sqrt(self.v)))
        for i in xrange(kern_dim_minus_one):
            print_info("\nSearching for sub kernel basis No. %d:" % (i+1))
            C = zeros(basis.shape, 'd')
            C[sub_dims, :] = basis[sub_dims, :]

            C = hstack((C, ones((self.v, 1), 'd')))
            P =asmatrix(svd(C)[0])[:, i+2:]

            # questions: 1. why need to add the all-1 vector again
            # (otherwise, all eigenvalues are NaN);

            # 2. why won't single triangle work in this case (cannot
            # find all components)?
    
            J = asmatrix(zeros((self.v, self.v), 'd'))
            for v in sub_dims:
                J[v, v] = 1.0

            J_bar = P.T * J * P
            S_bar = P.T * self.sigma * P

            val, vec = eig(a=S_bar, b=J_bar)
            
            #print "ev: ", val[:10]
            k = None
            mins = 1e100
            besty = None
            for j, v in enumerate(val):
                if abs(imag(v)) > eps or isnan(real(v)) or isnan(imag(v)):
                    continue
                
                y0 = asmatrix(vec)[:, j]
                n = norm((P * y0)[sub_dims,:])
                if n > eps:

                    y = y0/n
                    
                    ss = (y.T * S_bar * y)[0,0]
                    if  ss < mins:
                        print_info("\tnorm(J P y0)=%g\n\teigval=%s\n\teigvec_is_complex=%s\n\ty^t S_bar y = %s" % (n, str(v), str(norm(imag(y0)) > eps), str(ss)))
                        
                        mins = ss
                        k = j
                        besty = y
            
            if k == None:
                return zeros((len(sub_dims), kern_dim_minus_one), 'd'), zeros((kern_dim_minus_one))
                raise NoValidEigenValueFound()

            basis = hstack((basis, P * besty))

        return basis[sub_dims,1:], zeros((kern_dim_minus_one))

def enumerate_tris(adj):
    v = len(adj)
    tris = []
    in_list = set([])
    for u in xrange(v):
        for i, w in adj[u]:
            for j, x in adj[u]:
                if w == x:
                    continue
                for k, y in adj[w]:
                    if y != x:
                        continue
                    order = [u, w, x]
                    order.sort()
                    ordinal = order[0] + order[1] * v + order[2] * v * v
                    if not (ordinal in in_list):
                        in_list.add(ordinal)
                        tris.append([u, w, x])
    return tris

def detect_LC_from_kernel(g, kernel_basis, eps = 1e-4):
    print_info("Detecting linked components:")
    if g.d != 2:
        print_info("\tNon-2d: assume entire graph is connected for now...")
        return[range(g.v)]
    print_info("\tkernel_basis.shape = %s" % str(kernel_basis.shape))
    tris = enumerate_tris(g.adj)
    v_cc = [set([]) for i in xrange(g.v)]
    cc = []

    vec_1 = asmatrix(ones((g.v, 1),'d'))
    vec_1 /= norm(vec_1)
    s = kernel_basis.shape[1]
    kernel_basis = svd(kernel_basis - vec_1 * vec_1.T * kernel_basis)[0][:, :s]
    kernel_basis = asmatrix(hstack([vec_1, kernel_basis]))

    for t in tris:
        if len(v_cc[t[0]]) > 0 and len(v_cc[t[1]]) > 0 and len(v_cc[t[2]]) > 0:
            continue
        if matrix_rank(kernel_basis[t,:], eps) < g.d + 1:
            continue
        cn = len(cc)
        cc.append([])
        for v in t:
            v_cc[v].add(cn)
        cc[cn].extend(t)

        b = matrix(kernel_basis[t,:]).T
        b = asmatrix(svd(b)[0][:, :g.d+1])
        b_bT = asmatrix(b) * asmatrix(b.T)

        for u in xrange(g.v):
            if u in t or len(v_cc[u]) > 0:
                continue
            r = matrix_rank(kernel_basis[t+[u],:], eps*0.1)
            if r <= g.d+1:
                cc[cn].append(u)
                v_cc[u].add(cn)
#            else:
#                print "invalid rank: ", r

        while 1:
            to_remove = []
            for u in cc[cn]:
                deg = 0
                for edge, dest in g.adj[u]:
                    if cn in v_cc[dest]:
                        deg = deg + 1
                if deg < g.d:
                    to_remove.append(u)
            for u in to_remove:
                cc[cn].remove(u)
                v_cc[u].remove(cn)

            if len(to_remove) == 0:
                break

        print_info("\t%d: seed %s: size = %d" %
                   (cn, str(t), len(cc[cn])))

    for u in xrange(g.v):
        if len(v_cc[u]) == 0:
            cc.append([u])
                   
    return cc


def sample_kernel(g, ss):
    v, d, E = g.v, g.d, g.E
    ns = ss.shape[1]

    ass = zeros((v,v), 'd')
    for i in xrange(ns):
        w = ss[:,i]
        o = asmatrix(matrix_from_vector(w, E, v))
        ass += o.T * o

    return Kernel(ass)

#     if auto_dim_K:
#       return Kernel(as).
#     kernel(as, None, eps)
#     else:
#         if S.STRESS_SAMPLE == 'semilocal':
#             mdk = g.gsr.dim_K - 1
#         else:
#             mdk = g.gr.dim_K - 1
        
#         #return kernel(as, int(mdk * S.SDP_SAMPLE), eps)
#         return kernel(as, max(int(g.d * S.SDP_SAMPLE_MAX), mdk), eps)
#         #return kernel(as, v-1, eps)
#         #return kernel(as, mdk + int(g.d * (S.SDP_SAMPLE-1)), eps)
#         #return kernel(as, mdk, eps)
