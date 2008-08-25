from numpy import *
from scipy.linalg.basic import *
from util import *

class Graph:
    def __init__(self, p, E):
        self.p = p
        self.E = E
        self.v = p.shape[1]
        self.d = p.shape[0]
        self.e = len(E)
        self._adj = None

    def _get_adj(self):
        if self._adj == None:
            adj = [[] for i in xrange(self.v)]
            for i in xrange(self.e):
                u, w = self.E[i,0], self.E[i,1]
                adj[u].append((i, w))
                adj[w].append((i, u))

            for a in adj:
                a.sort(key=lambda x: x[1])

            self._adj = adj

        return self._adj
    
    adj = property(_get_adj)
    
    def connected_components(self):
        """ Return connected components of the graph.

        Return (sizes, cc_index) where sizes[i] is the size of the
        i-th connected component. Vertex v belongs to the
        cc_index[v]-th connected component.
        """
        visited = - ones((self.v), 'i')
        q = zeros((self.v), 'i')
        cc = []
        for i, a in enumerate(self.adj):
            if visited[i] >= 0:
                continue
            cc.append(0)
            visited[i] = c = len(cc)-1
            p = 1
            j = 0
            q[0] = i
            while j < p:
                u = q[j]
                j = j + 1
                for e, w in self.adj[u]:
                    if visited[w] < 0:
                        visited[w] = c
                        cc[c] = cc[c] + 1
                        q[p] = w
                        p = p + 1
        return array(cc), visited

    def get_k_ring_subgraphs(self, k, min_neighbor):
        """
        return k_rings of each vertex. If the #v of the ring is less than
        min_neighbor, additional vertices in larger rings are added till
        at least min_neighbor of them is in the graph. Result is stored
        as two list, one vtx indices, one edge indices.
        """
        print_info("Computing %d-ring subgraphs" % k)
        g = self
        v, adj = g.v, g.adj
        vtx_indices = [set([]) for i in xrange(v)]
        edge_indices = [set([]) for i in xrange(v)]
        anv = 0
        ane = 0
        min_neighbor = min(min_neighbor, v)
        for i in xrange(v):
            # do a length limited BFS:
            a = [(i, 0)]
            p = 0
            vtx_indices[i].add(i)
            prev_ring = set([i])
            cur_ring = 0
            while cur_ring < k or len(vtx_indices[i]) < min_neighbor:
                cur_ring = cur_ring + 1
                new_ring = set([])
                for w in prev_ring:
                    for edge, dest in adj[w]:
                        if not (dest in vtx_indices[i]):
                            vtx_indices[i].add(dest)
                            new_ring.add(dest)
                prev_ring = new_ring


            for w in vtx_indices[i]:
                for edge, dest in adj[w]:
                    if dest in vtx_indices[i]:
                        edge_indices[i].add(edge)

            anv = anv + len(vtx_indices[i])
            ane = ane + len(edge_indices[i])

            vtx_indices[i] = array(list(vtx_indices[i]), 'i')
            edge_indices[i] = array(list(edge_indices[i]), 'i')

        print_info("\taverage #v = %d, average #e = %d" % (anv/v, ane/v))
        return vtx_indices, edge_indices

    def subgraph_edges(self, v_subset):
        ridx = -ones((self.v), 'i')
        for i, w in enumerate(v_subset):
            ridx[w] = i

        t = array([[i, ridx[e[0]], ridx[e[1]]]
                   for i, e in enumerate(self.E)
                   if ridx[e[0]] >= 0 and ridx[e[1]] >= 0], 'i')

        if len(t) == 0:
            return zeros((0,2),'i'), zeros((0,),'i')
        return t[:,1:], t[:,0].ravel()
    
def subgraph(g, v_idx):
    return Graph(g.p[:,v_idx], g.subgraph_edges(v_idx)[0])


def largest_cc(g):
    print_info("Get largest connected component.")
    cc, cn = g.connected_components()
    if len(cc) == 1:
        return g
    c = argmax(cc)
    return subgraph(g, [i for i in xrange(g.v) if cn[i] == c])

def filter_dangling_v(g):
    while 1:
        ov = g.v
        adj = g.adj
        g = subgraph(g, [i for i in xrange(g.v) if len(adj[i]) > g.d])
        if g.v == ov:
            break
    return g

