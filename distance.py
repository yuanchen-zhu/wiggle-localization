#!/usr/bin/python
from numpy import *
from scipy.linalg.basic import *
from scipy.linalg.decomp import *
import sys, pickle, scipy.stats
from pylab import *
from matplotlib.collections import LineCollection
from geo import *
from util import *

EPS = 1e-12

def print_info(s):
    print "\t\t\t\t\tINFO:",  s

def build_edgeset(p, max_dist, min_dist):
    """
    Given a d x n numpyarray p encoding n d-dim points, returns a k x
    2 integer numpyarray encoding k edges whose two ends are distanced
    less than max_dist apart.
    """
    def inbetween (v, l0, l1):
        return v >= l0 and v <= l1

    V = xrange(p.shape[1])
    return array([[i,j] for i in V for j in V 
                  if i < j and inbetween(norm(p[:,i] - p[:,j]), min_dist, max_dist)], 'i')

def L(p, E, noise_std):
    """
    p is d x n and encodes n d-dim points. E is k x 2 and encodes the
    edge set. Returns k-dim column vector containing the squared
    distances of the edges.
    """
    v = asarray(p[:, E[:,0]] - p[:, E[:,1]])
    d = sqrt((v * v).sum(axis=0))
    noise = random.randn(len(E)) * noise_std
    d += noise
    return asmatrix(d * d).T

def locally_rigid_rank(v, d):
    return v * d - chooses(d+1,2)

def random_p(v, d):
    return asmatrix(array(random.random((d,v)), order='FORTRAN'))

def random_graph(v, d, dist_ratio, min_dist, discardNonrigid = True):
    i = 0
    while True:
        p = random_p(v, d)
        E = build_edgeset(p, dist_ratio * math.sqrt(d), min_dist)
        e = len(E)
        i = i + 1
        if discardNonrigid:
            t = locally_rigid_rank(v, d)
            if t >= e:
                print "Too few degress of freedom. Enlarge edge set!"
                return
            rig =  rigidity(v, d, E)
            if rig != "G":
                continue
        if discardNonrigid:
            print_info("Globally rigid graph created after %d tries" % i)

        print_info("|V|=%d |E|=%d" % (v, e))
        return p, E

def v_d_e_from_graph(g):
    p, E = g
    return p.shape[1], p.shape[0], len(E)

def rigidity(v, d, E, eps = EPS, rigidity_iter = 2, stress_iter = 2):
    t = locally_rigid_rank(v, d)
    e = len(E)
    rigidity_rank = 0
    stress_kernel_dim = e

    for x in xrange(rigidity_iter):
        p = random_p(v, d)
        D = zeros((len(E), d * v))
        for k in xrange(len(E)):
            i, j = E[k][0], E[k][1]
            diff_ij = p[:,i] - p[:,j]
            D[k, i*d:i*d+d] = 2.0 * diff_ij.T
            D[k, j*d:j*d+d] = -2.0 * diff_ij.T
        u, s, vh = svd(D)
        rigidity_rank = max(rigidity_rank, len(s[s >= eps]))

        if rigidity_rank != t:
            continue

        stress_basis = u[:,t:]
        for y in xrange(stress_iter):
            w = stress_basis * asmatrix(random.random((e - t, 1)))
            w /= norm(w)
            omega = stress_matrix_from_vector(w, E, v)
            eigval, eigvec = eig(omega)
            abs_eigval = abs(eigval)
            stress_kernel_dim = min(stress_kernel_dim, len(abs_eigval[abs_eigval <= eps]))

        if rigidity_rank == t and stress_kernel_dim == d + 1:
            break
    
    if rigidity_rank != t:
        return "N"
    elif stress_kernel_dim != d + 1:
        return "L"
    else:
        return "G"

def stress_matrix_from_vector(w, E, v):
    O = asmatrix(zeros((v,v), 'd'))
    V = xrange(v)
    for i in xrange(len(E)):
        p, q = E[i][0], E[i][1]
        O[p,q] = O[q,p] = w[i]
        O[p,p] -= w[i]
        O[q,q] -= w[i]
    return O

class Conf:
    pass

def dump_graph(file, g):
    p, E = g
    v, d, e = v_d_e_from_graph(g)

    conf = Conf()
    conf.v = v
    conf.d = d
    conf.E = E
    conf.p = p
    pickle.dump(conf, file)

class DOFTooFew(Exception):
    pass

class NotLocallyRigid(Exception):
    pass

class NotGloballyRigid(Exception):
    pass

def plot_info(name_param, cov_spec, t, stress_spec, d, p, approx_p, v, E):
    margin = 0.05
    width = (1.0 - margin * 4.0) / 3.0
    height = 1.0 - margin * 2.0

    figure(figsize=(12,4))
    clf()

    #Graph the spectral distribution of the covariance matrix
    axes([margin, margin, width, height])
    semilogy()
    plot(xrange(len(cov_spec)),cov_spec)
    axvline(t)
    axis([0, 300, 0.001, 1000])
    gca().set_aspect('auto')

    #Graph the spectral distribution of the stress matrix
    axes([margin*2+width, margin, width, height])
    loglog()
    plot(xrange(1, len(stress_spec)), stress_spec[1:])
    axvline(d)
    axis([1, v, 1e-8, 1e0])
    gca().set_aspect('auto')

    #Graph the geometry
    axes([margin*3+width*2, margin, width, height])
    scatter(x = approx_p[0].A.ravel(), y = approx_p[1].A.ravel(), s = 32, linewidth=(0.0), c = "r", marker = 'o', zorder=100)
    scatter(x = p[0].A.ravel(), y = p[1].A.ravel(), s = 32, linewidth=(0.0), c = "b", marker = 'o', zorder =101)
    axis([-0.2, 1.2, -0.2, 1.2])
    gca().set_aspect('equal')

    gca().add_collection(LineCollection([p.T[e] for e in E], colors = "lightgrey"))
    gca().add_collection(LineCollection([(approx_p.T[i].A.ravel(), p.T[i].A.ravel()) for i in xrange(v)], colors = "green"))

    savefig('infoplot-%s.png' % name_param)

def graph_scale(g, kernel_ratio, noise_std, sampling_ratio):
    p, E = g
    v, d, e = v_d_e_from_graph(g)
    
    # Sanity check: is the graph globally rigid
    t = locally_rigid_rank(v, d)
    if t >= e: raise DOFTooFew()

    rig =  rigidity(v, d, E)
    if rig == "N": raise NotLocallyRigid()
    elif rig == "L": raise NotGloballyRigid()

    # Now esimate the tangent plane
    N_samples = int(t * sampling_ratio)
    print_info("#samples = %d" % N_samples)
    
    #DL_of_deltas = asmatrix(zeros((e, N_samples), 'd'))

    #for i in range(N_samples):
    #    delta = (asmatrix(random.random((d,v))) - 0.5) * (kernel_ratio * 2)
    #    for j in range(n_measure):
    #        DL_of_deltas[:,i] += noisy_L(p + delta, E) - noisy_L(p, E)
    #    DL_of_deltas[:,i] /= n_measure

    DL_of_deltas = asmatrix(zeros((e, N_samples + 1), 'd'))
    DL_of_deltas[:,0] = L(p, E, noise_std)
    for i in xrange(N_samples):
        delta = asmatrix(random.random((d,v))) - 0.5
        delta *= (kernel_ratio*2)
        DL_of_deltas[:,i+1] = L(p + delta, E, noise_std)
    
    #mean = DL_of_deltas.mean(axis=1)
    #for i in range(N_samples+1):
    #    DL_of_deltas[:,i] -= mean

    u, s, vh = svd(DL_of_deltas)
    T_of_p_basis = u[:,:t]
    S_of_p_basis = u[:,t:]

    # Try to find a good stress matrix (i.e., with a clear difference
    # between the d'th and d+1'th eigenvalue (0-based indexing)
    stress_tries = 0
    maxratio = 0.0
    while stress_tries < 5 and stress_tries < e-t:
        #w = S_of_p_basis * asmatrix(random.random((e - t,1)))
        w = S_of_p_basis[:, -stress_tries-1]
        w /= norm(w)
        omega = stress_matrix_from_vector(w, E, v)
        eigval, evec = eig(omega)

        abs_eigval = abs(eigval)

        order =  range(v)
        order.sort(key = lambda i: abs_eigval[i])

        r = abs_eigval[order[d + 1]] / abs_eigval[order[d]]
        if r > maxratio:
            maxratio = r
            eigvec = evec[:, order]
            abseig = abs_eigval[:, order]

        if r > 20.0:
            break
        stress_tries = stress_tries + 1

    
    print_info("eigval[d+1]/eigval[d] = %g" % maxratio)

    q = asmatrix(vstack([(eigvec[:,i]).T for i in xrange(d)]))
    A = optimal_affine(q, p)
    approx_p = (A * homogenous_vectors(q))[:d,:]
    diff = p - approx_p 


    # plot the info
    plot_info(name_param = 'noise-%.06f-ker-%.06f-sampling-%.04d'
              % (noise_std, kernel_ratio, sampling_ratio),
              cov_spec = s, t = t,
              stress_spec = abseig, d = d,
              p = p, approx_p = approx_p, v = v, E = E)

    error = norm(diff)
    avg_error = math.sqrt(error * error / v)
    print_info("Average per-point error: %g\n" % avg_error)
    sys.stdout.flush()
    return avg_error

def graph_scale_directional(g, kernel_ratio, noise_std, sampling_ratio):
    p, E = g
    v, d, e = v_d_e_from_graph(g)

def main():
    random.seed(0)
    v = 50
    d = 2
    ratio = 0.2
    n_tests = 1
    

    print "#V = %d  D = %d  dist_ratio = %g  n_tests = %d " % (v, d, ratio, n_tests)
    print "%12s; %12s; %12s; %12s; %12s" % ("perturb", "noise dev", "samp. ratio", "err. mean", "err. dev") 

    #for noise_std in map(lambda x:0.0001 * pow(2,x/2.0) - 0.0001, xrange(16)):
    #    for kern in map(lambda x:0.0001 * pow(2,x/2.0), xrange(16)):
    #for noise_std in map(lambda x:0.00001 * pow(2,x/1.5) - 0.00001, xrange(16)):
    #    for kern in map(lambda x:0.05/16.0 * x+0.0001, xrange(17)):
    #for noise_std in [0.0000001]:
    g = random_graph(v, d, ratio, 0.01)
    for noise_std in [0, 0.001]:
        for sampling_ratio in [2, 4, 8, 16, 32]:
            for kern in map(lambda x:0.05/16.0 * (x)+0.0001, [6]):
                e = graph_scale(g = g, 
                                kernel_ratio=kern,
                                noise_std = noise_std, 
                                sampling_ratio = sampling_ratio)

                print "%12.8f; %12.8f; %12.8f; %12.8f" % (kern, noise_std, sampling_ratio, e)
                sys.stdout.flush()
        print ""


if __name__ == "__main__":
    main()
