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

class Kernel:
    def __init__(self, sigma):
        self.v = sigma.shape[0]
        self.sigma = asmatrix(sigma)
        self.eigval, self.eigvec = eigh(self.sigma)
        
    def extract(self, eps = S.EPS):
        kd = len(self.eigval[self.eigval < eps])
        return self.eigvec[:,1:kd], self.eigval[1:]

    def extract_sub(self, sub_dims, kern_dim_minus_one, eps = S.EPS):

        basis = asmatrix(ones(((self.v), 1), 'd')* (1.0/sqrt(self.v)))
        for i in xrange(kern_dim_minus_one):
            C = zeros(basis.shape, 'd')
            C[sub_dims, :] = basis[sub_dims, :]

            if len(sub_dims) < self.v:
                C = hstack((C, ones((self.v, 1), 'd')))

            u, s, vt = svd(C)
            if len(sub_dims) < self.v:
                P = asmatrix(u)[:, i+2:]
            else:
                P = asmatrix(u)[:, i+1:]

            new_sigma = P.T * self.sigma * P
            if False: #method 1
                
                val, vec = eigh(new_sigma)
                
                print_info "ev: ", val[:10]
                ns = [norm((P * asmatrix(vec)[:, j])[sub_dims,:]) for j in xrange(len(val))]
                print "n: ", ns[:10]
                k = None
                for j in xrange(len(val)):
                    if ns[j] > 1e-1:
                        k = j
                        break
                
                if k == None:
                    k = 0
            else: #method 2

                J = asmatrix(zeros((self.v, self.v), 'd'))
                for v in sub_dims:
                    J[v, v] = 1.0

                J = P.T * J * P
                S = P.T * self.sigma * P

                val, vec = eig(a=S, b=J)
                
                #print "ev: ", val[:10]
                k = None
                mins = 1e100
                for j in xrange(len(val)):
                    if abs(imag(val[j])) > eps or isnan(imag(val[j])) or isnan(real(val[j])):
                        continue
                    
                    w = asmatrix(vec)[:, j]
                    n = norm((P * w)[sub_dims,:])
                    if n > eps:
                        print n, val[j], vec[:,j]
                        
                        y = w/n
                        print norm(S * w), norm(J * w)
                        
                        ss = y.T * S * y
                        if  ss < mins:
                            mins = ss
                            k = j
                
                if k == None:
                    k = 0

            print k, val[k], vec[:, k]
            y = P * real(asmatrix(vec)[:, k])
            basis = hstack((basis, y / norm(y[sub_dims,:])))


        B = basis[sub_dims,:]
        print B.T * B

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
            if matrix_rank(kernel_basis[t+[u],:], eps*0.1) <= g.d+1:
                cc[cn].append(u)
                v_cc[u].add(cn)

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
