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
    sys.stderr.write("INFO:%s\n" % s)
    sys.stderr.flush()

def build_edgeset(p, max_dist, min_dist):
    """
    Given a d x n numpyarray p encoding n d-dim points, returns a k x
    2 integer numpyarray encoding k edges whose two ends are distanced
    less than max_dist apart.
    """
    def inbetween (v, l0, l1):
        return v >= l0 and v <= l1

    V = xrange(p.shape[1])
    return array(
        [[i,j] for i in V for j in V if i < j and
         inbetween(norm(p[:,i] - p[:,j]), min_dist, max_dist)], 'i')

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

class DOFTooFew(Exception):
    pass

class NotLocallyRigid(Exception):
    pass

class NotGloballyRigid(Exception):
    pass


def random_graph(v, d, max_dist, min_dist, discardNonrigid = True):
    i = 0
    while 1:
        p = random_p(v, d)
        E = build_edgeset(p, max_dist, min_dist)
        e = len(E)
        i = i + 1
        if discardNonrigid:
            t = locally_rigid_rank(v, d)
            if t >= e:
                raise DOFTooFew()
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
            stress_kernel_dim = min(stress_kernel_dim,
                                    len(abs_eigval[abs_eigval <= eps]))

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

figure(figsize=(12,8))

def plot_info(stats, cov_spec, t, stress_spec, d,
              p, approx_p, approx_p2, v, E):
    margin = 0.05
    width = 1.0/3.0
    height = (1.0-margin*2)/2.0

    clf()

    axes([0,0,1,1-margin*2], frameon=False, xticks=[], yticks=[])
    title('Noise:%.04f   Kern.:%.04f   Sampling:%2.02f'
          % (stats["noise"], stats["perturb"], stats["sampling"]))
    figtext(width*2-margin, height*2-margin*1.5,
            'aff. err: %.06f\naff. L-err: %.06f'
            % (stats["af g error"], stats["af l error"]),
            ha = "right", va = "top", color = "green")
    figtext(width*3-margin*2, height*2-margin*1.5,
            'err: %.06f\nL-err: %.06f' %
            (stats["l g error"], stats["l l error"]),
            ha = "right", va = "top", color = "red")
    
    
    #Graph the spectral distribution of the covariance matrix
    axes([margin, margin, width-margin*2, height-margin*2])
    loglog()
    plot(xrange(1, 1+len(cov_spec)),cov_spec)
    axvline(t+1)
    axis([1, min(t*16, len(E))+1, 1e-4, 1e1])
    gca().set_aspect('auto')
    title("Cov. Spec.")

    #Graph the spectral distribution of the stress matrix
    axes([margin, height+margin, width-margin*2, height-margin*2])
    loglog()
    plot(xrange(1, 1 + len(stress_spec)), stress_spec)
    axvline(d)
    axvline(d+1)
    axis([1, 1+len(stress_spec), 1e-2, 1e2 ])
    gca().set_aspect('auto')
    title("Agg. Stress Kern. Spec.")

    #Graph the geometry
    axes([margin+width, margin, width*2-margin*2, height*2-margin*2])
    title("Error")
    scatter(x = approx_p2[0].A.ravel(), y = approx_p2[1].A.ravel(), s = 32,
            linewidth=(0.0), c = "green", marker = 'o', zorder=99, alpha=0.75)
    scatter(x = approx_p[0].A.ravel(), y = approx_p[1].A.ravel(), s = 32,
            linewidth=(0.0), c = "r", marker = 'o', zorder=100, alpha=0.75)
    scatter(x = p[0].A.ravel(), y = p[1].A.ravel(), s = 32,
            linewidth=(0.0), c = "b", marker = 'o', zorder =102, alpha=1)
    axis([-0.2, 1.2, -0.2, 1.2])
    gca().set_aspect('equal')

    gca().add_collection(LineCollection([
        p.T[e] for e in E], colors = "lightgrey", alpha=0.75))
    gca().add_collection(LineCollection([
        (approx_p2.T[i].A.ravel(), p.T[i].A.ravel())
        for i in xrange(v)], colors = "green", alpha=0.75))
    gca().add_collection(LineCollection([
        (approx_p.T[i].A.ravel(), p.T[i].A.ravel())
        for i in xrange(v)], colors = "red", alpha=0.75))

    fn = 'infoplot-n%.04f-k%.04f-s%2.02f.png' % (
        stats["noise"], stats["perturb"], stats["sampling"])
    print_info("%s dumped" % fn)
    savefig(fn)

def check_ggr(g):
    p, E = g
    v, d, e = v_d_e_from_graph(g)
    
    # Sanity check: is the graph generic globally rigid
    dim_T = locally_rigid_rank(v, d)
    if dim_T >= e:
        raise DOFTooFew()
    else:
        rig =  rigidity(v, d, E)
        if rig == "N": raise NotLocallyRigid()
        elif rig == "L": raise NotGloballyRigid()

def measure_L_rho(g, perturb, noise_std, n_samples):
    p, E = g
    v, d, e = v_d_e_from_graph(g)

    print_info("#measurements = %d" % n_samples)

    #L_rhos = asmatrix(zeros((e, n_samples), 'd'))

    #for i in range(N_samples):
    #    delta = (asmatrix(random.random((d,v))) - 0.5) * (perturb * 2)
    #    for j in range(n_measure):
    #        L_rhos[:,i] += noisy_L(p + delta, E) - noisy_L(p, E)
    #    L_rhos[:,i] /= n_measure

    L_rhos = asmatrix(zeros((e, n_samples), 'd'))
    for i in xrange(n_samples):
        delta = asmatrix(random.random((d,v))) - 0.5
        delta *= (perturb*2)
        L_rhos[:,i] = L(p + delta, E, noise_std)

    #mean = L_rhos.mean(axis=1)
    #for i in range(n_samples+1):
    #    L_rhos[:,i] -= mean

    return L_rhos, L(p, E, noise_std)

def estimate_stress_space(L_rhos, dim_T):
    u, s, vh = svd(L_rhos)
    S_basis = asmatrix(u[:,dim_T:])
    return S_basis, s

def estimate_stress_space2(L_rhos, dim_T):
    pass

def estimate_stress_kernel(g, S_basis):
    p, E = g
    v, d, e = v_d_e_from_graph(g)
    
    n_S = S_basis.shape[1]

    ss_samples = min(n_S, 300)

    # If you want randomize:
    stress_space = svd(asmatrix(S_basis) *
                       asmatrix(random.random((n_S, ss_samples+ss_samples/2))))[0][:,:ss_samples]
    # Else take the last ss_samples stress vectors in S_basis
    #stress_space = S_basis[n_S-ss_samples:]

    n_per_matrix = d*4 # number of basis vectors to pick from kernel of each matrix
    stress_kernel = zeros((v, ss_samples * n_per_matrix))

    stress_mul = ones(ss_samples)
    # for i in xrange(stress_space.shape[1]):
    #    j = t + i
    #    if j >= len(s):
    #        j = len(s) - 1
    #    stress_mul[i] = -math.log(s[j])

    print_info("stress space samples = %d" % ss_samples)
    for i in xrange(ss_samples):
        w = stress_space[:, i]
        omega = stress_matrix_from_vector(w, E, v)
        eigval, eigvec = eig(omega)
        eigval = abs(eigval)
    
        order =  range(v)
        order.sort(key = lambda i: eigval[i])
        stress_kernel[:, i*n_per_matrix : (i+1)*n_per_matrix] = eigvec[:,order[1:n_per_matrix+1]] * stress_mul[i]

    stress_kernel_basis, ev, whatever = svd(stress_kernel)
    return stress_kernel_basis[:, :d], ev

def calculate_relative_positions(g, L_rho, q):
    B = optimal_linear_transform_for_l(q, g[1], L_rho)
    return B * q

def graph_scale(g, perturb, noise_std, sampling):
    p, E = g
    v, d, e = v_d_e_from_graph(g)

    check_ggr(g)

    dim_T = locally_rigid_rank(v, d)
    n_samples = int(dim_T * sampling)

    L_rhos, L_rho = measure_L_rho(g, perturb, noise_std, n_samples)
    S_basis, cov_spec = estimate_stress_space(L_rhos, dim_T)
    K_basis, stress_spec = estimate_stress_kernel(g, S_basis)
    
    q = asmatrix(K_basis.T) # coordinate vector

    T_q = calculate_relative_positions(g, L_rho, q)

    l_approx = (optimal_rigid(T_q, p) * homogenous_vectors(T_q))[:d,:]
    af_approx = (optimal_affine(q, p) * homogenous_vectors(q))[:d,:]


    l = L(p, E, 0)

    def mean_l2_error(v1, v2):
        d, n = v1.shape
        return mean(array([norm(v1[:,i] - v2[:,i]) for i in xrange(n)]))
    
    af_g_error = mean_l2_error(p, af_approx)
    af_l_error = mean_l2_error(l.T, L(af_approx, E, 0).T)

    l_g_error = mean_l2_error(p, l_approx)
    l_l_error = mean_l2_error(l.T, L(l_approx, E, 0).T)
    
    # plot the info
    plot_info({"noise":noise_std,
               "perturb":perturb,
               "sampling":sampling,
               "l g error": l_g_error,
               "l l error": l_l_error,
               "af g error": af_g_error,
               "af l error": af_l_error},
              cov_spec = cov_spec, t = dim_T,
              stress_spec = stress_spec, d = d,
              p = p, approx_p = l_approx, approx_p2 = af_approx, v = v, E = E)

    return af_g_error

def main():
    random.seed(0)
    v = 200
    d = 2
    k =  math.pow(2*(d+1), 1.0/d)/3.0
    dist_threshold = 3*k*math.pow(v, -1.0/d)
    n_tests = 1
    
    print "#V = %d  D = %d  max_dist = %g  n_tests = %d " % (
        v, d, dist_threshold, n_tests)

    g = random_graph(v, d, dist_threshold, 0.01)
    for noise_std in array([0.01, 0.02]) * dist_threshold:
        for perturb in array([5, 10]) * noise_std:
            for sampling in [1.5, 2, 4, 8]:
                e = graph_scale(g = g, 
                                perturb=max(1e-4, perturb),
                                noise_std = noise_std,
                                sampling = sampling)
                sys.stdout.flush()
            if noise_std == 0:
                break


if __name__ == "__main__":
    main()
