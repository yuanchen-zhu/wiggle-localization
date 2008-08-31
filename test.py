import settings as S*
from util import *
from stress import *
from genrig import *
from graph import *
from dist import *
from numpy import *
from scipy.linalg.basic import *
from scipy.linalg.decomp import *

v=8
d=2
E=array([[0,1],[0,2],[1,2],[1,4],[2,3],[2,5],[3,4],[3,7],[4,5],[5,6],[5,7],[6,7],[3,6],[0,4]], 'i')
e=len(E)
gr = GenericRigidity(v, d, E)

g = Graph(random_p(v, d, None), E)
g.gr = gr

dim_T = locally_rigid_rank(v, d)
n_samples = int(dim_T * 16)

L_rhos, L_rho = measure_L_rho(g, 1e-5, 0, n_samples)

S_basis, cov_spec = estimate_stress_space(L_rhos, dim_T)
stress_samples = sample_from_stress_space(S_basis)
K_basis, stress_spec = estimate_stress_kernel(g, stress_samples)

vcc, cc = detect_linked_components_from_stress_kernel(g, K_basis)
print vcc, cc

## D = rigidity_matrix(v, d, E)
## t = matrix_rank(D)
## sb = asmatrix(svd(D)[0][:,t:])
## print e
## print t
## print sb.shape
## w = sb * asmatrix(random.random((e-t,1)))
## #w = sb[:,0]
## w /= norm(w)

## omega = stress_matrix_from_vector(w, E, v)
## eigval, eigvec = eig(omega)
## eigval = abs(eigval)

## order =  range(v)
## order.sort(key = lambda i: eigval[i])
## print eigval[order]

## skd = len(eigval[eigval <= EPS])

## K = eigvec[:,order[:skd]]

    
    

