from numpy import *
from scipy.linalg.basic import *
from scipy.linalg.decomp import *
import settings as S
import cPickle

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
    global info_buffer
    info_buffer.append(s)

def get_sim_params_and_stats(hash_str):
    f = open("%s/%s.result" % (S.DIR_PLOT, hash_str), "rb")
    p = cPickle.load(f)             # the original param
    stats = cPickle.load(f)
    f.close()
    return p, stats

def check_info(params, output_params, override_graph):
    """ Check if [hash].info corresponding to the current settings
    exist and print if it does."""

    try:
        fn = "%s/%s.info" % (S.DIR_PLOT, get_settings_hash(params, override_graph))
        f = open(fn, "rb")
        l = f.read()
        print "### Cached info %s present. Will skip actural run." %fn
        if not output_params.SKIP_PRINT_OLD:
            print l
        f.close()

        fn = "%s/%s.result" % (S.DIR_PLOT, get_settings_hash(params, override_graph))
        f = open(fn, "rb")
        p = cPickle.load(f)             # the original param
        stats = cPickle.load(f)
        f.close()

        import os
        mtime_stamp = os.path.getmtime(fn)

        import datetime
        mtime = datetime.datetime.fromtimestamp(mtime_stamp)

        stats.mtime = mtime
        return stats
    except IOError:
        return None
    except EOFError:
        return None

def flush_info(params, stats, graph_override):
    """ Flush info to [hash].info where hash is hash(settings) and write error to [hash].result"""
    h = get_settings_hash(params, graph_override)
    global info_buffer
    f = open("%s/%s.info" % (S.DIR_PLOT, h), "wb")
    for l in info_buffer:
        f.write(l)
        f.write('\n')
    f.close()
    info_buffer = []
    f = open("%s/%s.result" % (S.DIR_PLOT, h), "wb")
    import cPickle
    cPickle.dump(params, f)
    cPickle.dump(stats, f)
    f.close()

def dump_settings(params):
    import settings
    ss = get_module_consts(settings)
    print_info("Settings: ")
    for s in ss:
        print_info("\t%s :  %s" % (s[0], s[1]))
    print_info("Parameters: ")
    for s in get_object_consts(params):
        print_info("\t%s :  %s" % (s[0], s[1]))


def flush_info(params, stats, graph_override):
    """ Flush info to [hash].info where hash is hash(settings) and write error to [hash].result"""
    h = get_settings_hash(params, graph_override)
    global info_buffer
    f = open("%s/%s.info" % (S.DIR_PLOT, h), "wb")
    for l in info_buffer:
        f.write(l)
        f.write('\n')
    f.close()
    info_buffer = []
    f = open("%s/%s.result" % (S.DIR_PLOT, h), "wb")
    import cPickle
    cPickle.dump(params, f)
    cPickle.dump(stats, f)
    f.close()

def dump_settings(params):
    import settings
    ss = get_module_consts(settings)
    print_info("Settings: ")
    for s in ss:
        print_info("\t%s :  %s" % (s[0], s[1]))
    print_info("Parameters: ")
    for s in get_object_consts(params):
        print_info("\t%s :  %s" % (s[0], s[1]))


def get_settings_hash(params, override_graph):
    import settings
    if not override_graph:
        h = hash((tuple(get_object_consts(params)), tuple(get_module_consts(settings))))
    else:
        h = hash((tuple(get_object_consts(params)), tuple(get_module_consts(settings)), tuple(override_graph)))

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
            m = m + random.uniform(-1e-20, 1e-20, m.shape)

def matrix_rank(m, eps = S.EPS, printS = False):
    u, s, vh = svd_conv(m)
    if printS:
        print s
    return len(s[abs(s) > eps])
          
        
def random_p(v, d, sample_space_pred):
    p = asmatrix(zeros((d,v), order='FORTRAN'))
    for i in xrange(v):
        done = 0
        while not done:
            q = asmatrix(random.uniform(size=(d))).T
            if sample_space_pred == None or sample_space_pred(q):
                done = 1
        p[:,i] = q
    return p

def get_module_consts(mod):
    return [(n, mod.__getattribute__(n)) for n in dir(mod) if n.isupper()]

def get_object_consts(obj):
    return [(n, getattr(obj, n)) for n in dir(obj) if n.isupper()]

    
def ridx(idx, size):
    r = -ones((size), 'i')
    for i, w in enumerate(idx):
        r[w] = i
    return r
