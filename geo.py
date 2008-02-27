from numpy import *
from scipy.linalg.basic import *
from scipy.linalg.decomp import *
import sys, pickle, scipy.stats, pylab

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
