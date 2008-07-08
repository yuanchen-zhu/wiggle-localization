from numpy import *
from settings import *
from util import *
from scipy.linalg.basic import *

def stress_matrix_from_vector(w, E, v):
    O = asmatrix(zeros((v,v), 'd'))
    V = xrange(v)
    for i in xrange(len(E)):
        p, q = E[i][0], E[i][1]
        O[p,q] = O[q,p] = w[i]
        O[p,p] -= w[i]
        O[q,q] -= w[i]
    return O

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

def detect_linked_components_from_stress_kernel(g, kernel_basis, eps = 2e-3):
    tris = enumerate_tris(g.adj)
    v_cc = [set([]) for i in xrange(g.v)]
    cc = []
    vec_1 = asmatrix(ones((g.v, 1),'d'))
    vec_1 /= norm(vec_1)
    s = kernel_basis.shape[1]
    kernel_basis = svd(kernel_basis - vec_1 * vec_1.T * kernel_basis)[0][:, :s]
    kernel_basis = asmatrix(hstack([vec_1, kernel_basis]))
    for t in tris:
        if len(v_cc[t[0]] & v_cc[t[1]] & v_cc[t[2]]) > 0:
            continue
        cn = len(cc)
        cc.append([])
        for v in t:
            v_cc[v].add(cn)
        cc[cn].extend(t)

        b = matrix(kernel_basis[t,:].T)
        b = svd(b)[0][:,:3]
        b_bT = asmatrix(b) * asmatrix(b.T)

        print_info("seed %s" % str(t))
        
        for u in xrange(g.v):
            if u in t:
                continue
            x = kernel_basis[u,:].T
            n = norm(x - b_bT * x)
            if n < eps:
                cc[cn].append(u)
                v_cc[u].add(cn)
                #print_info("\t adding %d (residule = %g)" % (u, n))
            else:
                #print_info("\t discarding %d (residule = %g)" % (u, n))
                pass

        print_info("linked component %d: rank = %d" % (
            cn, matrix_rank(kernel_basis[cc[cn],:], 1e-2)))
                          

    return cc
