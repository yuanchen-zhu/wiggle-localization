from numpy import *
import settings as S
from util import *
from scipy.linalg.basic import *
import sys
import genrig

def estimate_space(Ls, dim_T):
    u, s, vh = svd(Ls)              # dense
    return u[:,dim_T:], s

def calculate_exact_space(g):
    D = genrig.rigidity_matrix(g.v, g.d, g.E, g.p)
    u, s, vh = svd_conv(D)               # E by dv, sparse, 2dE non-zero
    t = len(s[s >= S.EPS])
    return u[:,t:], s

            
def sample(S_basis, params):
    n_S = S_basis.shape[1]
    nss = min(n_S, params.SS_SAMPLES)

    print_info("Get random stresses...")

    if params.RANDOM_STRESS:
        ss = asmatrix(S_basis) * asmatrix(random.random((n_S, nss+nss/2)))
        if params.ORTHO_SAMPLES:
            if ss.shape[0] != 0 and ss.shape[1] != 0:
                ss = svd(ss)[0]
        return ss[:,:nss]
    else:
        # take the last SS_SAMPLES stress vectors in S_basis
        return S_basis[:,n_S-params.SS_SAMPLES:]

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
        
    def extract(self, d, eps = S.EPS):
        kd = len(self.eigval[self.eigval < eps])
        kd = max(kd, d+1)
        return self.eigvec[:,1:kd], self.eigval[1:]

    def extract_sub(self, lcs, cur_lc_id, kern_dim_minus_one, eps = S.EPS):
        sub_dims = lcs[cur_lc_id]
        
        if len(sub_dims) == 1:
            return zeros((1, kern_dim_minus_one), 'd'), zeros((1,1), 'd')

        #SS = matrix(self.sigma)
        #for i in xrange(self.v):
        #    if not (i in lcs):
        #        SS[i,:] = 0
        #        SS[:,i] = 0
                

        #for i in xrange(self.v):
        #    SS[i,i] = sum(SS[i, 
                

        basis = asmatrix(ones(((self.v), 1), 'd'))
        for i in xrange(kern_dim_minus_one):
            #print_info("\nSearching for sub kernel basis No. %d:" % (i+1))
            

            J = asmatrix(zeros((self.v, self.v), 'd'))
            for v in sub_dims:
                J[v, v] = 1.0

            C0 = zeros(basis.shape, 'd')
            C0[sub_dims, :] = basis[sub_dims, :]


            norm_J = norm(J)
            norm_sigma = norm(self.sigma)

            if norm_J > eps:
                nJ = J / norm_J
            else:
                nJ = J
            if norm_sigma > eps:
                nsigma = self.sigma / norm_sigma
            else:
                nsigma = self.sigma

            #s, u = eigh(nJ.T * nJ + nsigma.T * nsigma)
            s, u = eigh(nJ + nsigma)
            C1 = asmatrix(u)[:, abs(s) < eps]

            C = hstack((C0, C1))
            #print_info("dim(C)=%s" % str(C.shape))
            

            if C.shape[1] > 0:
                u, s, vt = svd(C)
                P = asmatrix(u)[:, [i for i in xrange(self.v) if i >= len(s) or abs(s[i]) < eps]]
            else:
                P = asmatrix(identity(self.v))

            # questions V1: 1. why need to add the all-1 vector again
            # (otherwise, all eigenvalues are NaN);

            # 2. why won't single triangle work in this case (cannot
            # find all components)?

            # Answer: cannot have S_bar and J_bar have common kernel,
            # or return anaything as eigenvalues, and some guy in the
            # common kernel as eigen vector.
            #
            # questions V2: 1. why won't single triangle work?
            # 2. scaling issues: why does the first two coordinate not
            # good even in the no noise case.
            
            J_bar = P.T * J * P
            S_bar = P.T * self.sigma * P
            if J_bar.shape[0] == 0 or J_bar.shape[1] == 0 or S_bar.shape[0] == 0 or S_bar.shape[1] == 0:
                break
            J_bar_rank = matrix_rank(J_bar)
            S_bar_rank = matrix_rank(S_bar)

            #print_info("dim(J_bar)=dim(S_bar)=%s" % str(J_bar.shape))
            #print_info("rank(J_bar)=%d" % J_bar_rank)
            #print_info("rank(S_bar)=%d" % S_bar_rank)

            val, vec = eig(a=S_bar, b=J_bar)
            
            #print "ev: ", val[:10]
            k = None
            mins = 1e100
            besty = None

            n_imag_discarded = 0
            n_zero_norm_discarded = 0
            n_picked = 0
            n_discarded = 0
            for j, v in enumerate(val):
                if abs(imag(v)) > eps: #or isnan(real(v)) or isnan(imag(v)):
                    n_imag_discarded = n_imag_discarded + 1
                    continue
                
                y0 = asmatrix(vec)[:, j]
                n = norm((P * y0)[sub_dims,:])
                if n > eps:

                    y = (P * y0)/n
                    y0 = zeros(y.shape, 'd')
                    y0[sub_dims,:] = y[sub_dims, :]
                    
                    #ss = abs((y0.T * self.sigma * y0)[0,0])
                    ss = abs((y.T * self.sigma * y)[0,0])
                    if  ss < mins:
                        #print_info("Picked:\n\tnorm(J P y0)=%g\n\tnorm(S_bar y0)=%g\n\teigval=%s\n\teigvec_is_complex=%s\n\ty^t S_bar y = %s"
                        #           % (n, abs(norm(S_bar * y0)), str(v), str(norm(imag(y0)) > eps), str(ss)))

                        n_picked = n_picked + 1
                        mins = ss
                        k = j
                        besty = y
                    else:
                        n_discarded = n_discarded + 1
                        #print_info("Discarded:\n\tnorm(J P y0)=%g\n\tnorm(S_bar y0)=%g\n\teigval=%s\n\teigvec_is_complex=%s\n\ty^t S_bar y = %s"
                        #           % (n, abs(norm(S_bar * y0)), str(v), str(norm(imag(y0)) > eps), str(ss)))

                        
                else:
                    n_zero_norm_discarded = n_zero_norm_discarded + 1
                    #print_info("Zero norm, Discarded:\n\tnorm(J P y0)=%g\n\tnorm(S_bar y0)=%g\n\teigval=%s\n\teigvec_is_complex=%s"
                    #           % (n, abs(norm(S_bar * y0)), str(v), str(norm(imag(y0)) > eps)))
                     

            #print_info("Picked: %d\tDiscarded: %d\tImaginary: %d\t Zero Norm: %d" % (n_picked, n_discarded, n_imag_discarded, n_zero_norm_discarded))
            if k == None:
                break

            basis = hstack((basis, besty))

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
