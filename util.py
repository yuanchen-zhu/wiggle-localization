from numpy import *
from scipy.linalg.basic import *
from scipy.linalg.decomp import *
from settings import *

def factorial(n):              
    """
    Returns n!
    """
    p = 1
    for j in range(1,n+1):
        p *= j
    return p

def chooses(n, r):
    """
    Return n chooses r, or n!/r!/(n-r)!
    """
    return factorial(n)/factorial(r)/factorial(n-r)

def print_info(s):
    import sys
    sys.stderr.write("INFO: %s\n" % s)
    sys.stderr.flush()

def svd_conv(m):
    for i in xrange(10):
        try:
            u, s, vh = svd(m)
            return u, s, vh
        except LinAlgError:
            m = m + random.randn(m.shape[0], m.shape[1]) * 1e-20

def matrix_rank(m, eps = EPS):
    u, s, vh = svd(m)
    return len(s[s > eps])
          
        
def random_p(v, d, sample_space_pred):
    p = asmatrix(zeros((d,v), order='FORTRAN'))
    for i in xrange(v):
        done = 0
        while not done:
            q = asmatrix(random.random((d))).T
            if sample_space_pred == None or sample_space_pred(q):
                done = 1
        p[:,i] = q
    return p

def get_module_consts(mod):
    return [(n, mod.__getattribute__(n)) for n in dir(mod) if n.isupper()]

    
