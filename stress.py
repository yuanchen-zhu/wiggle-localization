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
    eigval, eigvec = eig(omega)     # v by v, sparse, 2vd non-zero entries
    eigval = abs(eigval)

    order =  range(v)
    order.sort(key = lambda i: eigval[i])

    if kern_dim_minus_one == None:
        kern_dim_minus_one = len(eigval[eigval < eps]) - 1

    kd = kern_dim_minus_one + 1

    kern = eigvec[:,order[:kd]]

    # for accuracy, remove vec_1 component
    kern = svd(kern - vec_1_op * kern)[0][:,:kd-1]
    return kern, eigval[order]


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

def detect_linked_components_from_stress_kernel(g, kernel_basis, eps = EPS):
    print_info("Detecting linked components:")
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

        print_info("\t%d: seed %s: rank = %d" %
                   (cn, str(t), matrix_rank(kernel_basis[cc[cn],:], EPS * 10)))
                          

    return cc


def sample_stress_kernel(g, ss):
    v, d, E = g.v, g.d, g.E
    ns = ss.shape[1]
    
    nks = min((g.gr.dim_K - 1) * KERNEL_SAMPLES, v)
    print_info("Taking %d eigenvectors from each stress matrix" % nks)
    ks = zeros((v, ns * nks))

    print_info("Computing kernel for %d stress" % ns)

    for i in xrange(ns):
        w = ss[:, i]
        o = stress_matrix_from_vector(w, E, v)
        kern, oev = calculate_single_stress_kernel(o, nks)
        
        k = i*nks
        ks[:, k:k+nks] = kern
        
        if WEIGHT_KERNEL_SAMPLE:
            ks[:, k: k+nks] *= -log(oev[:nks])

        sys.stdout.write('.')
        sys.stdout.flush()

    sys.stdout.write('\n')
    print_info("Calculating dominant stress kernel...")

    #return ks, [0]

    K_basis, ev, whatever = svd(ks) # v by C, where C = nks * ns, dense
    return K_basis[:,:g.gr.dim_K -1], ev

