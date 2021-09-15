from numpy import *
from scipy.linalg.basic import *

ROTATE_BY_90 = matrix([[0, -1],
                       [1, 0]], 'd')

def trilaterate2d(u, r):
    """
    u is a 2x3 matrix encoding 3 vertex position in 2d.  r is 3
    element array encoding distances between the unknown point to the
    three known points. The coord of the unknown point is returned as
    a 3x1 matrix (i.e., column vector)
    """
    u = asmatrix(u)
    u0 = u[:, 0]
    u1 = u[:, 1]
    u2 = u[:, 2]

    # Denote the points as u0, u1, u2. Set u0 as the origin. Set u0u1
    # as the x axis. The y axis is just the x axis rotated 90 degrees
    # counterclockwise. Next get the coords of u1 = (d, 0), and u2 =
    # (i, j) in the local frame.

    x = u1 - u0
    d = norm(x)
    x /= d
    y = ROTATE_BY_90 * x

    u0u2 = u2-u0
    i = (x.T * u0u2)[0,0]
    j = (y.T * u0u2)[0,0]

    xx = (r[0] * r[0] - r[1] * r[1] + d * d) / (2 * d)
    yy = (r[0] * r[0] - r[2] * r[2] + i * i + j * j) / (2 * j) - (i / j) * xx

    return u0 + xx * x + yy * y


def localize_given_seeds(adj, r, seed_tri, seed_tri_pos):
    v = len(adj)
    if r != None:
        p = asmatrix(zeros((2, v), 'd'))
        p[:,seed_tri] = seed_tri_pos
    else:
        p = None

    # flag = 0 => vertex not localized, not in queue
    # flag = 1 => vertex not localized, but in queue
    # flag = 2 => vertex localized
    flag = zeros((v), dtype=int8)

    queue = []
    queue.extend(seed_tri)
    flag[queue] = [2, 2, 2]
    q = 0
    used_edges = []
    while q < len(queue):
        u = queue[q]
        q = q + 1
        if flag[u] != 2:
            # see if we can localize
            anchors = []
            anchor_edges = []
            for i, w in adj[u]:
                if flag[w] == 2:
                    anchors.append(w)
                    anchor_edges.append(i)
                    if len(anchors) >= 3:
                        break
            if len(anchors) >= 3:
                # yes! can localize
                if r != None:
                    p[:,u] = trilaterate2d(p[:,anchors], r[anchor_edges])
                flag[u] = 2
                used_edges.extend([[u, anchors[0]], [u, anchors[1]], [u, anchors[2]]])

        if flag[u] != 2:
            flag[u] = 0
        else:
            # check its neighbors and maybe add to the queue
            for i, w in adj[u]:
                if flag[w] == 0:
                    flag[w] = 1
                    queue.append(w)

    localized = [i for i in xrange(v) if flag[i] == 2]
            
    return p, array(localized, 'i'), array(used_edges, 'i')


def position_tri(r):
    x = (r[2] * r[2] - r[1] * r[1])/(2 * r[0]) + r[0]/2
    y = math.sqrt(r[2] * r[2] - x * x)
    return matrix([[0, 0], [r[0], 0], [x, y]]).T


def enumerate_tris(adj, r):
    v = len(adj)
    tris = []
    tri_edges = []
    in_list = set([])
    for u in xrange(v):
        for i, w in adj[u]:
            for j, x in adj[u]:
                if w == x:
                    continue
                for k, y in adj[w]:
                    if y != x:
                        continue
                    order = [u, w, x]
                    order.sort()
                    ordinal = order[0] + order[1] * v + order[2] * v * v
                    if not (ordinal in in_list):
                        in_list.add(ordinal)
                        tris.append([u, w, x])
                        if r != None:
                            tri_edges.append([r[i], r[k], r[j]])
    return tris, tri_edges

class TriResult:
    pass

# r is the distance, if None, then only find largest trilateration graph
def trilaterate_graph(adj, r = None):
    tris, tri_edge_len = enumerate_tris(adj, r)
    max_num_localized = 0
    tri = TriResult()
    tri.p = None
    tri.localized = array([], 'i')
    tri.used_edges = array([], 'i')
    tri.seed = [-1, -1, -1]
    for i, seed_tri in enumerate(tris):
        if r != None:
            seed_pos = position_tri(tri_edge_len[i])
        else:
            seed_pos = None

        p, localized, used_edges = localize_given_seeds(adj, r, seed_tri, seed_pos)
        if len(localized) > max_num_localized:
            max_num_localized = len(localized)
            tri.p = p
            tri.localized = localized
            tri.used_edges = used_edges
            tri.seed = seed_tri
            if max_num_localized == len(adj):
                break

    return tri
