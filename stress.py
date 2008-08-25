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


vec_1_op = None

def calculate_single_stress_kernel(omega, kern_dim_minus_one = None, eps = EPS):
    global vec_1_op
    v = omega.shape[0]
    if vec_1_op == None or vec_1_op.shape[0] != v or vec_1_op.shape[1] != v:
        vec_1_op = asmatrix(ones((v, v), 'd') / float(v))
    
    # The following uses dense matrix routines
    if SK_USE_SVD:
        u, s, vh = svd(omega)
        if kern_dim_minus_one == None:
            kern_dim_minus_one = len(s[s < eps]) - 1
        kd = kern_dim_minus_one + 1
        return asmatrix(vh).T[:,v-kd:v-1], s
    else:
        eigval, eigvec = eig(omega)     # v by v, sparse, 2vd non-zero entries
        eigval = abs(eigval)

        order =  range(v)
        order.sort(key = lambda i: eigval[i])

        if kern_dim_minus_one == None:
            kern_dim_minus_one = len(eigval[eigval < eps]) - 1

        kd = kern_dim_minus_one + 1

        # for accuracy, remove vec_1 component
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

    if not PER_LC_KS_PCA:
        S = zeros((v,v), 'd')
        for i in xrange(ns):
            w = ss[:,i]
            o = asmatrix(stress_matrix_from_vector(w, E, v))
            S += o.T * o

        k, e= calculate_single_stress_kernel(S, int((g.gr.dim_K - 1)))
        return k, e #array([e[len(e)-1-i] for i in xrange(len(e))])
    else:
        nks = min(int((g.gr.dim_K - 1) * KERNEL_SAMPLES), int(KERNEL_MAX_RATIO * v))
        print_info("Taking %d eigenvectors from each stress matrix" % nks)

        F = 1
        ks = asmatrix(zeros((v, ns * nks * F)))
        print_info("Computing kernel for %d stress" % ns)

        for i in xrange(ns):
            w = ss[:, i]
            o = stress_matrix_from_vector(w, E, v)
            kern, oev = calculate_single_stress_kernel(o, nks)

            k = i * nks * F
            if F == 1:
                a = range(kern.shape[1])
                a.reverse()
                ks[:, k:k+nks] = kern[:, a]
                if WEIGHT_KERNEL_SAMPLE:
                    ks[:, k: k+nks*F] *= -log(oev[:nks])
            else:
                rc = asmatrix(random.random((kern.shape[1], nks * F)))
                for j in xrange(rc.shape[1]):
                    rc[:,j] /= norm(rc[:,j])
                ks[:, k:k+nks * F] = asmatrix(kern) * rc

            sys.stdout.write('.')
            sys.stdout.flush()

        sys.stdout.write('\n')
        print_info("Calculating dominant stress kernel...")

        if PER_LC_KS_PCA:
            return ks, [0]
        else:
            if ns == 1:
                K_basis, ev = ks, oev
            else:
                K_basis, ev, whatever = svd(ks) # v by C, where C = nks * ns, dense

            return K_basis[:,:int((g.gr.dim_K - 1) * SDP_SAMPLE)], ev

