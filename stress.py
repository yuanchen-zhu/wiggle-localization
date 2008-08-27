from numpy import *
from settings import *
from util import *
from scipy.linalg.basic import *
import sys

def stress_matrix_from_vector(w, E, v):
    O = asmatrix(zeros((v,v), 'd'))
    V = xrange(v)
    for i in xrange(len(E)):
        p, q = E[i][0], E[i][1]
        O[p,q] = O[q,p] = w[i]
        O[p,p] -= w[i]
        O[q,q] -= w[i]
    return O


def calculate_single_stress_kernel(omega, kern_dim_minus_one = None, eps = EPS):
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

def detect_linked_components_from_stress_kernel(g, kernel_basis, eps = 1e-9):
    print_info("Detecting linked components:")
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


def sample_stress_kernel(g, ss):
    v, d, E = g.v, g.d, g.E
    ns = ss.shape[1]

    S = zeros((v,v), 'd')
    for i in xrange(ns):
        w = ss[:,i]
        o = asmatrix(stress_matrix_from_vector(w, E, v))
        S += o.T * o

    return calculate_single_stress_kernel(S, max(int(g.d * SDP_SAMPLE), g.gr.dim_K - 1))
