from numpy import *
from scipy.linalg.basic import *
from scipy.linalg.decomp import *

def translate_matrix(t):
    """
    Given a d-dim vector t, returns (d+1)x(d+1) matrix M such that
    left multiplication by M on a homogenuous (d+1)-dim vector v
    translate v by t (assuming the last coordinate of v is 1).
    """
    t = asarray(t).ravel()
    d = len(t)
    m = identity(d+1)
    m[:d,d] = t[:]
    return m

def homogenous_matrix(m):       
    """
    Expand a dxd matrix m to a (d+1)x(d+1) matrix m' such that the
    last the last row and column of m' are all zero except at the last
    coordinate.
    """
    d = m.shape[0]
    n = zeros((d+1, d+1))
    n[:d,:d] = m[:,:]
    n[d,d] = 1
    return asmatrix(n)

def homogenous_vectors(v):
    """
    Stick an extra 1 at the end of v, increasing its dimension by 1.
    """
    return vstack((v, asmatrix(ones(v.shape[1]))))

def optimal_affine(p, q):
    """
    Find the optimal affine transformation M minimizing ||Mp - q||
    """
    p_star = p.mean(axis=1)
    q_star = q.mean(axis=1)
    pp = p - p_star
    qq = q - q_star
    M = (qq * pp.T) * inv(pp * pp.T)
    return translate_matrix(q_star) * homogenous_matrix(M) * translate_matrix(-p_star)

def optimal_rigid(p, q):
    p_star = p.mean(axis=1)
    q_star = q.mean(axis=1)
    pp = p - p_star
    qq = q - q_star
    U, s, V = svd(asmatrix(qq) * asmatrix(pp.T))
    M = asmatrix(U) * asmatrix(V)
    return translate_matrix(q_star) * homogenous_matrix(M) * translate_matrix(-p_star)

def optimal_linear_transform_for_l(p, E, l):
    """
    E is an edge set (set of pairs) and l assigns a real number to
    each edge in E. This function finds the optimal linear
    transformation M such that the sum over {i,j} in E of the squared
    difference between l({i,j}) and the squared length of the
    difference between the i-th and j-th column of Mp"""
    
    d, v = p.shape
    n = d * (d+1)/2
    A = zeros((E.shape[0], n), 'd')

    diff = asarray(p[:, E[:, 0]] - p[:, E[:, 1]]).T
    i = 0
    for r in xrange(d):
        for c in xrange(r + 1):
            if r == c:
                A[:, i] = diff[:, r] * diff[:, r]
            else:
                A[:, i] = 2 * diff[:, r] * diff[:, c]
            i = i + 1

    # solve least square A x = l
    x = lstsq(A, l)[0]

    S = zeros((d, d))
    i = 0
    for r in xrange(d):
        for c in xrange(r + 1):
            if r == c:
                S[r, c] = x[i]
            else:
                S[r, c] = S[c, r] = x[i]
            i = i + 1

    # find M s.t transpose(M) * M = S
    u, s, v = svd(S)
    return asmatrix(diag(sqrt(s), 0)) * asmatrix(v)
    

    
