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

info_buffer = []

def print_info(s):
    import sys
    sys.stderr.write("INFO: %s\n" % s)
    sys.stderr.flush()
    info_buffer.append(s)

def check_info():
    """ Check if [hash].info corresponding to the current settings
    exist and print if it does."""

    try:
        fn = "%s/%s.info" % (DIR_PLOT, get_settings_hash())
        f = open(fn, "rb")
        l = f.read()
        print "### Cached info %s present. Will skip actural run." %fn
        print l
        f.close()
        return True
    except IOError:
        return False

def flush_info():
    """ Flush info to [hash].info where hash is hash(settings)"""
    f = open("%s/%s.info" % (DIR_PLOT, get_settings_hash()), "wb")
    for l in info_buffer:
        f.write(l)
        f.write('\n')
    f.close()

def dump_settings():
    import settings
    ss = get_module_consts(settings)
    print_info("Settings: ")
    for s in ss:
        print_info("\t%s :  %s" % (s[0], s[1]))

def get_settings_hash():
    import settings
    h = hash(tuple(get_module_consts(settings)))
    if h < 0:
        return '0'+str(-h)
    else:
        return str(h)

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

    
