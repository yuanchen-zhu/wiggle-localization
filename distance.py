#!/usr/bin/python
from numpy import *
from scipy.linalg.basic import *
from scipy.linalg.decomp import *
import sys, pickle, scipy.stats, pylab
from geo import *
from util import *

EPS = 1e-12

def degeneratePos(p, threshold):
    V = range(p.shape[1])
    for i in V:
        for j in range(i+1, p.shape[1]):
            if norm(p[:,i] - p[:,j]) < threshold:
                return True
    return False

def inbetween (v, l0, l1):
    return v >= l0 and v <= l1

def buildE(p, distThreshold, smallThreshold=0.05):
    """
    Given a d x n numpyarray p encoding n d-dim points, returns a k x
    2 integer numpyarray encoding k edges whose two ends are distanced
    less than distThreshold apart.
    """
    V = range(p.shape[1])
    return array([[i,j] for i in V for j in V 
                  if i < j and inbetween(norm(p[:,i] - p[:,j]), smallThreshold, distThreshold)], 'i')

def L(p, E, noise_dev = 0.0):
    """
    p is d x n and encodes n d-dim points. E is k x 2 and encodes the
    edge set. Returns k-dim column vector containing the squared
    distances of the edges.
    """
    v = asarray(p[:, E[:,0]] - p[:, E[:,1]])
    d = sqrt((v * v).sum(axis=0))
    noise = random.randn(len(E)) * noise_dev
    d += noise
    return asmatrix(d * d).T

def locally_rigid_rank(v, d):
    return v * d - chooses(d+1,2)

def rigidity(v, d, E, eps = EPS, rigidity_iter = 2, stress_iter = 2):
    t = locally_rigid_rank(v, d)
    e = len(E)
    rigidity_rank = 0
    stress_kernel_dim = e
    for x in range(rigidity_iter):
        p = random_v(v, d)
        D = zeros((len(E), d * v))
        for k in range(len(E)):
            i, j = E[k][0], E[k][1]
            diff_ij = p[:,i] - p[:,j]
            D[k, i*d:i*d+d] = 2.0 * diff_ij.T
            D[k, j*d:j*d+d] = -2.0 * diff_ij.T
        u, s, vh = svd(D)
        rigidity_rank = max(rigidity_rank, len(s[s >= eps]))

        if rigidity_rank != t:
            continue

        stress_basis = u[:,t:]
        for y in range(stress_iter):
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
    V = range(v)
    for i in range(len(E)):
        p, q = E[i][0], E[i][1]
        O[p,q] = O[q,p] = w[i]
        O[p,p] -= w[i]
        O[q,q] -= w[i]
    return O


def random_v(v, d, discardClose = True):
    while (True):
        p = asmatrix(array(random.random((d,v)), order='FORTRAN'))
        #if discardClose and degeneratePos(p, 0.1):
        #    continue
        return p

def random_graph(v, d, dist_ratio, discardClose = True, discardNonrigid = True):
    while True:
        p = random_v(v, d, discardClose = discardClose);
        E = buildE(p, dist_ratio * math.sqrt(d))
        e = len(E)
        if discardNonrigid:
            t = locally_rigid_rank(v, d)
            if t >= e:
                print "Too few degress of freedom. Enlarge edge set!"
                return
            rig =  rigidity(v, d, E)
            if rig != "G":
                continue
        return p, E

class Conf:
    pass


def print_configuration(file, v, d, E, p):
    conf = Conf()
    conf.v = v
    conf.d = d
    conf.E = E
    conf.p = p
    pickle.dump(conf, file)

def test_random(v, d, dist_ratio, kernel_ratio, noise_dev = 0.0, sampling_ratio = 2.0, niter = 1, n_measure = 2, g = None):
    saveout = sys.stdout
    sys.stdout = open("log", "ab")
    locally_rigid_conf = open("locally-rigid", "ab")
    non_rigid_conf = open("none-rigid", "ab")
    globally_rigid_invalid_stress = open("globally-rigid-invalid-stress", "ab")

    iter = 0
    errors = []
    if g != None:
        GEN_GRAPH = False
        p, E = g
        e = len(E)
    else:
        GEN_GRAPH = True
    V = range(v)

    while iter < niter:

        print "@NEW CONFIGURATION\n#v = %d  d = %d  edge_dist_ratio = %g  perturb = %g  measure_dev = %g  sampling_ratio = %g" % (v, d, dist_ratio, kernel_ratio, noise_dev, sampling_ratio)
    
        print "Creating p and E"

        if GEN_GRAPH:
            p = random_v(v, d);
            E = buildE(p, dist_ratio * math.sqrt(d))
            e = len(E)
    
        print "|V| = " + str(v) + " |E| = " + str(e)
    
        t = locally_rigid_rank(v, d)
        if t >= e:
            print "Too few degress of freedom. Enlarge edge set!"
            return

        rig =  rigidity(v, d, E)
        if rig == "N":
            print "None rigid configuration. Trying with new random dataset.\n"
            print_configuration(non_rigid_conf, v, d, E, p)
            continue
        elif rig == "L":
            print "Locally but not globally rigid. Trying with new random dataset.\n"
            print_configuration(locally_rigid_conf, v, d, E, p)
            continue
        else:
            print "Globally rigid"
            iter = iter + 1
            
        def noisy_L(p, E):
            return L(p, E, noise_dev)

        N_samples = int(t * sampling_ratio)
        print "Creating " + str(N_samples) + " perturbations..."
        
        #DL_of_deltas = asmatrix(zeros((e, N_samples), 'd'))
        #for i in range(N_samples):
        #    delta = (asmatrix(random.random((d,v))) - 0.5) * (kernel_ratio * 2)
        #    for j in range(n_measure):
        #        DL_of_deltas[:,i] += noisy_L(p + delta, E) - noisy_L(p, E)
        #    DL_of_deltas[:,i] /= n_measure
        DL_of_deltas = asmatrix(zeros((e, N_samples + 1), 'd'))
        DL_of_deltas[:,0] = noisy_L(p, E)
        for i in range(N_samples):
            delta = asmatrix(random.random((d,v))) - 0.5
            delta *= (kernel_ratio*2)
            DL_of_deltas[:,i+1] = noisy_L(p + delta, E)
        
        #mean = DL_of_deltas.mean(axis=1)
        #for i in range(N_samples+1):
        #    DL_of_deltas[:,i] -= mean
    
        print "Estimating T(p) and S(p)..."
        print DL_of_deltas.shape
        u, s, vh = svd(DL_of_deltas)
        print len(s)
        T_of_p_basis = u[:,:t]
        S_of_p_basis = u[:,t:]

        S_eigen_distr = s[t:] - s[t:].min()
        S_eigen_distr /= S_eigen_distr.max()
        pylab.clf()
        #pylab.hist(s[t:], bins=40, normed = 0)
        #pylab.hist(s[:], bins=40, normed = 0)
        pylab.semilogy()
        pylab.plot(range(len(s)),s)
        pylab.axvline(t)
        pylab.axis([0, 600, 0.001, 1000]) 
        pylab.savefig('cov-eigenvalue-noise-%.06f-ker-%.06f-sampling-%.04d-%02d.png' % (noise_dev, kernel_ratio, sampling_ratio, iter))

        #pylab.clf()
        #pylab.hist(S_eigen_distr, bins=40, normed = 1)
        #pylab.savefig('cov-eigenvalue-pdf-%.6f-ker-%.6f-sampling-%.4d-%d.png' % (noise_dev, kernel_ratio, sampling_ratio, iter))
    
        print "Eigen decomposing stress matrix..."

        stress_tries = 0
        maxratio = 0.0
        tt = 0
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
                eigval_order = order
                eigvec = evec
                abseig = abs_eigval

                tt = tt + 1
                print abs_eigval[order[0:10]]

            if r > 20.0:
                break
            stress_tries = stress_tries + 1

        pylab.clf()
        pylab.semilogy()
        pylab.plot(range(len(abseig)-1), abseig[eigval_order[1:]])
        pylab.axvline(d-1)
        pylab.axis([0, 50, 1e-8, 1e0]) 
        pylab.savefig('stress-ev-noise-%.06f-ker-%.06f-sampling-%.04d-%02d.png' % (noise_dev, kernel_ratio, sampling_ratio, iter))
            
        print "Random stress matrix tried: %d" % (stress_tries+1)
        if stress_tries >= 10:
            print "Failed to find a stress matrix with rank %d nullspace" % (d + 1)
            print_configuration(globally_rigid_invalid_stress, v, d, E, p)
        
    
        eigval_order = filter(lambda i: var(eigvec[:,i]) > EPS, eigval_order) # remove constant
        
        print "Finding q..."
        q = asmatrix(vstack([(eigvec[:,eigval_order[i]]).T for i in range(d)]))
    
        print "Finding optimal affine transformation from p to q"
        A = optimal_affine(q, p)
        approx_p = (A * homogenous_vectors(q))[:d,:]
        diff = p - approx_p 

        pylab.clf()
        pylab.quiver(approx_p[0].A.ravel(), approx_p[1].A.ravel(), diff[0].A.ravel(), diff[1].A.ravel(),
                     color = "grey", units = "x", width=0.003, scale = 1, headwidth=4, headlength=8)
        pylab.scatter(x = approx_p[0].A.ravel(), y = approx_p[1].A.ravel(), s = 32, linewidth=(0.0), c = "r", marker = 'o')
        pylab.scatter(x = p[0].A.ravel(), y = p[1].A.ravel(), s = 32, linewidth=(0.0), c = "b", marker = 'o')
        pylab.axis([-0.2, 1.2, -0.2, 1.2])
        pylab.savefig('points-noise-%.06f-ker-%.06f-sampling-%.04d-%02d.png' % (noise_dev, kernel_ratio, sampling_ratio, iter))

    
        error = norm(diff)
        avg_error = math.sqrt(error * error / v)
    
        print "Average per-point error: %g\n" % avg_error

        errors.append(avg_error)

        sys.stdout.flush()


    sys.stdout.close()
    locally_rigid_conf.close()
    non_rigid_conf.close()
    globally_rigid_invalid_stress.close()

    sys.stdout = saveout
    return array(errors, 'd')

def main():
    pylab.figure(figsize=(8,8))
    v = 50
    d = 2
    ratio = 0.3
    n_tests = 1
    n_measure = 1
    

    print "#V = %d  D = %d  dist_ratio = %g  n_tests = %d n_measure = %d" % (v, d, ratio, n_tests, n_measure)
    print "%12s; %12s; %12s; %12s; %12s" % ("perturb", "noise dev", "samp. ratio", "err. mean", "err. dev") 

    #for noise_dev in map(lambda x:0.0001 * pow(2,x/2.0) - 0.0001, range(16)):
    #    for kern in map(lambda x:0.0001 * pow(2,x/2.0), range(16)):
    #for noise_dev in map(lambda x:0.00001 * pow(2,x/1.5) - 0.00001, range(16)):
    #    for kern in map(lambda x:0.05/16.0 * x+0.0001, range(17)):
    #for noise_dev in [0.0000001]:
    g = random_graph(v, d, ratio)
    for noise_dev in [0.001]:
        for kern in map(lambda x:0.05/16.0 * (x)+0.0001, [6, 12]):
            for sampling_ratio in [2, 4, 8, 16, 32]:
                e = test_random(v=v, d=d, g = g, dist_ratio=ratio, 
                                kernel_ratio=kern, noise_dev = noise_dev, 
                                sampling_ratio = sampling_ratio, 
                                n_measure = n_measure,
                                niter = n_tests)
                print "%12.8f; %12.8f; %12.8f; %12.8f; %12.8f" % (kern, noise_dev, sampling_ratio, e.mean(), e.std())
                sys.stdout.flush()
        print ""


if __name__ == "__main__":
    main()
