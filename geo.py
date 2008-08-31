from numpy import *
from scipy.linalg.basic import *
from scipy.linalg.decomp import *
from util import *
import settings as S

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
    return asmatrix(m)

def scale_matrix(t):
    """
    Given a d-dim vector t, returns (d+1)x(d+1) matrix M such that
    left multiplication by M on a homogenuous (d+1)-dim vector v
    scales v by t (assuming the last coordinate of v is 1).
    """
    t = asarray(t).ravel()
    d = len(t)
    m = identity(d+1)
    for i in xrange(d):
        m[i,i] = t[i]
    return asmatrix(m)
    

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
    M = (qq * pp.T) * pinv(pp * pp.T)
    return translate_matrix(q_star) * homogenous_matrix(M) * translate_matrix(-p_star)

def optimal_rigid(p, q):
    p_star = p.mean(axis=1)
    q_star = q.mean(axis=1)
    pp = p - p_star
    qq = q - q_star
    U, s, V = svd(asmatrix(qq) * asmatrix(pp.T))
    M = asmatrix(U) * asmatrix(V)
    return translate_matrix(q_star) * homogenous_matrix(M) * translate_matrix(-p_star)

def optimal_linear_transform_for_l_lsq(p, d, E, l):
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



def optimal_linear_transform_for_l_sdp(p, d, E, l):
    """
    E is an edge set (set of pairs) and l assigns a real number to
    each edge in E. This function finds the optimal linear
    transformation M such that the sum over {i,j} in E of the squared
    difference between l({i,j}) and the squared length of the
    difference between the i-th and j-th column of Mp"""

    k, v = p.shape
    n = k * (k+1)/2
    m = E.shape[0]
    A = zeros((m, n), 'd')

    diff = asarray(p[:, E[:, 0]] - p[:, E[:, 1]]).T
    i = 0
    for r in xrange(k):
        for c in xrange(r + 1):
            if r == c:
                A[:, i] = diff[:, r] * diff[:, r]
            else:
                A[:, i] = 2 * diff[:, r] * diff[:, c]
            i = i + 1

    # Minimize ||A x - l|| subjec to constraint symmatrix(x) is
    # semi-positive definite. This is equivalent to the following
    # semi-definite optimization:
    #
    # Minimize t
    # Subject to constraint
    #  (1) t >= |A x - l|^2
    #  (2) symmatrix(x) semi-pos definite
    #
    # Constraint (1):
    #   <=> (Ax-l)^T (Ax-l) - t <= 0
    #   <=> | I         Ax-l|  >= 0
    #       | (Ax-l)^T  t   |
    #
    # Constraint (2):
    #  <=> unpack x into a symmetric matrix S and S >= 0

    import cvxopt.coneprog;
    from cvxopt.base import matrix, spmatrix
    from cvxopt import solvers
    cvxopt.coneprog.options['DSDP_Monitor'] = 10
    cvxopt.coneprog.options['DSDP_GapTolerance'] = 1e-5
    cvxopt.coneprog.options['DSDP_MaxIts'] = 200
    
    rs, cs, vs = [], [], []
    for i in xrange(n):
        r = range(m) + [m] * m
        c = [m] * m + range(m)
        v = list(A[:,i].ravel())
        v = v + v
        for j in xrange(m*2):
            cs.append(i)
            rs.append(c[j] * (m+1) + r[j])
            vs.append(-v[j])
    cs.append(n)
    rs.append(m*(m+1)+m)
    vs.append(-1.0)
    G0 = spmatrix(vs, rs, cs, ((m+1)*(m+1), n+1))

    r = range(m) + range(m) + [m]*m
    c = range(m) + [m]*m + range(m)
    v = [1.0]*m + list(-l.A.ravel()) + list(-l.A.ravel())
    h0 = matrix(spmatrix(v, r, c, (m+1,m+1)))

    rs, cs, vs = [], [], []
    i = 0
    for r in xrange(k):
        for c in xrange(r + 1):
            if r == c:
                vs.append(-1.0)
                rs.append(c * k + r)
                cs.append(i)
            else:
                vs.extend([-1.0, -1.0])
                rs.extend([c *k + r, r *k + c])
                cs.extend([i, i])
            i = i + 1
    G1 = spmatrix(vs, rs, cs, (k*k, n+1))

    # Use (I * EPS) because the DSDP solver seems to enforce strict
    # inequality constraints, i.e., S < 0 instead of S <= 0.
    h1 = matrix(spmatrix([S.EPS]*k,range(k),range(k),(k,k)))

    c = matrix([[0.0]*n + [1.0]])
    
    if S.SDP_USE_DSDP:
        sol = solvers.sdp(c, Gs=[G0,G1], hs=[h0,h1], solver="dsdp")
    else:
        sol = solvers.sdp(c, Gs=[G0,G1], hs=[h0,h1])
    
    x = sol['x']

    s = zeros((k, k))
    i = 0
    for r in xrange(k):
        for c in xrange(r + 1):
            if r == c:
                s[r, c] = x[i]
            else:
                s[r, c] = s[c, r] = x[i]
            i = i + 1

    # find M s.t transpose(M) * M = rank_restricted(s, d)
    e, v = eig(s)                      # we have s = v * diag(e) * v.T
    e, v = sqrt(e.real), v.real
    order = range(k)
    order.sort(key = lambda i: -e[i]) # e[order] is sorted in descreasing value
    print_info("Transformation matrix eigenvalues: %s" % str(e[order]))
    print_info("Transformation matrix id: %s" % str(order))
    return asmatrix(dot(diag(e[order[:d]]), v.T[order[:d],:]))


def intersect2d(t, u, v, w):
    """
    Test if tu intersects vw. t, u, v, and w are 2x1 numpy matrix.
    """
    if min(t[0,0], u[0,0]) > max(v[0,0], w[0,0]):
        return
    if max(t[0,0], u[0,0]) < min(v[0,0], w[0,0]):
        return
    if min(t[1,0], u[1,0]) > max(v[1,0], w[1,0]):
        return
    if max(t[1,0], u[1,0]) < min(v[1,0], w[1,0]):
        return

    return (cross(w-v, t-v, axis=0) * cross(u-v, w-v, axis=0))[0] >= 0 and (cross(w-u, t-u, axis=0) * cross(t-u, v-u, axis=0))[0] >= 0

def inside_poly2d(x, poly):
    """
    Test if point x is inside the closed polygon poly. poly is a 2xn
    numpy matrix
    """
    nv = poly.shape[1]
    outp = poly.min(axis = 1) - matrix([100,100]).T

    # Test how many edges of the poly intersects outp-x
    n = 0
    for i in xrange(nv):
        if intersect2d(poly[:,i], poly[:,(i+1)%nv], x, outp):
            n = n + 1

    return n % 2 > 0
        
        
        
                                       
