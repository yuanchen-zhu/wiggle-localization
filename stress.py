from numpy import *
import settings as S
from util import *
from scipy.linalg.basic import *
import sys
import genrig

def estimate_stress_space(Ls, dim_T):
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


def kernel(omega, kern_dim_minus_one = None, eps = S.EPS):
    v = omega.shape[0]
    
    eigval, eigvec = eig(omega)     # v by v, sparse, 2vd non-zero entries
    eigval = abs(eigval)

    order =  range(v)
    order.sort(key = lambda i: eigval[i])

    if kern_dim_minus_one == None:
        kern_dim_minus_one = len(eigval[eigval < eps]) - 1

    kd = kern_dim_minus_one + 1

    return eigvec[:,order[1:kd]], eigval[order[1:]]

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
        print_info("\t%d: seed %s: size = %d, rank = %d" %
                   (cn, str(t), len(cc[cn]), matrix_rank(kernel_basis[cc[cn],:], eps)))

    for u in xrange(g.v):
        if len(v_cc[u]) == 0:
            cc.append([u])
                   
    return cc


def sample_kernel(g, ss, auto_dim_K = False, eps = S.EPS):
    v, d, E = g.v, g.d, g.E
    ns = ss.shape[1]

    as = zeros((v,v), 'd')
    for i in xrange(ns):
        w = ss[:,i]
        o = asmatrix(matrix_from_vector(w, E, v))
        as += o.T * o

    if auto_dim_K:
        return kernel(as, None, eps)
    else:
        if S.STRESS_SAMPLE == 'semilocal':
            mdk = g.gsr.dim_K - 1
        else:
            mdk = g.gr.dim_K - 1
        
        return kernel(as, max(int(g.d * S.SDP_SAMPLE), mdk), eps)
