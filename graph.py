from numpy import *
from scipy.linalg.basic import *
import sys

class Graph:
    pass

def v_d_e_from_graph(g):
    p, E = g
    return p.shape[1], p.shape[0], len(E)

def adjacency_list_from_edge_list(g):
    p, E = g
    v, d, e  = v_d_e_from_graph(g)
    adj = [[] for i in xrange(v)]
    for i in xrange(e):
        adj[E[i,0]].append((i, E[i,1]))
        adj[E[i,1]].append((i, E[i,0]))
    return adj

def build_edgeset(p, max_dist, min_dist, max_neighbors = None):
    """
    Given a d x n numpyarray p encoding n d-dim points, returns a k x
    2 integer numpyarray encoding k edges whose two ends are distanced
    less than max_dist apart, and each vertex has max number of
    neighbors <= max_neighbors.
    """
    def inbetween (v, l0, l1):
        return v >= l0 and v <= l1

    V = xrange(p.shape[1])

    if max_neighbors == None:
        return array(
            [[i,j] for i in V for j in V if i < j and
             inbetween(norm(p[:,i] - p[:,j]), min_dist, max_dist)], 'i')
    else:
        E = set([])
        for i in V:
            t = [(norm(p[:,i] - p[:,j]), j) for j in V if i != j]
            t = filter(lambda x: inbetween(x[0], min_dist, max_dist), t)
            t.sort(key = lambda x: x[0])
            for j in xrange(min(len(t), max_neighbors)):
                E.add((min(i,t[j][1]), max(i,t[j][1])))
        A = array(list(E), 'i')
        print len(A)
        return A

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

