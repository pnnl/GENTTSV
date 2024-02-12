from scipy.sparse import coo_array
import numpy as np
import networkx as nx
## some functions for importing and dealing with hypergraphs
def get_val(d, key):
    # to consistently relabel from 0
    if key not in d:
        n = len(d)
        d[key] = n
        return n
    return d[key]
    
def parse_hyper(name, min_edge_size = None, max_edge_size = None, get_node_labels = False, get_cats = False):
    """
    Parse in a hyperegraph from the text files.
    Parameters:
        name: name of the dataset (string)
        min_edge_size: the minimum size of hyperedge to keep. Must be at least s for the walk to run always. (int)
        max_edge_size: the maximum size of hyperedge to keep. (int)
        get_node_labels: if you want to also parse in the edge labels, make this statement true. (bool)
    Returns:
        node_dict a dictionary with nodes as keys and the hyperedge indeces they belong to as values. (dict)
        edge_dict: a dictionary with edge indexes as keys and the nodes that belong in them as values. (dict)
        node_labels: optional, a list of the labels of each node (list)
    """
    edge_dict = {}
    node_dict = {}
    node_map = {}
    edge_map = {}
    node_labels = {}
    label_map = {}
    file = "data/" + name + "/hyperedges.txt"
    full_edges = []
    with open(file) as f:
        for (e, line) in enumerate(f):
            edge = list(map(int, line.strip().split(',')))
            full_edges.append(edge)
            if max_edge_size != None:
                if len(edge) > max_edge_size:
                    continue
            if min_edge_size != None:
                if len(edge) < min_edge_size:
                    continue
            edge_id = get_val(edge_map, e)
            if edge_id not in edge_dict:
                edge_dict[edge_id] = []
            for node in edge:
                node_id = get_val(node_map, node)
                edge_dict[edge_id].append(node_id)
                if node_id not in node_dict:
                    node_dict[node_id] = [edge_id]
                else:
                    node_dict[node_id].append(edge_id)
    if get_cats:
        file = "data/" + name + "/hyperedge-" + 'labels' + ".txt"
        cats = np.loadtxt(file, dtype = int)
        n_cats = max(cats)
        cat_mat = np.zeros((len(node_map), n_cats))
        for (i, edge) in enumerate(full_edges):
            for node in edge:
                if node in node_map:
                    cat_mat[node_map[node], cats[i] - 1] += 1
        category = np.argmax(cat_mat, axis = 1)
    if get_node_labels:        
        file = "data/" + name + "/node-labels.txt"
        with open(file) as f:
            for (node, line) in enumerate(f):
                if node + 1 not in node_map:
                    continue
                label = line.strip()
                # label_id = get_val(label_map, label)
                # node_labels[node_map[node + 1]] = label_id
                node_labels[node_map[node + 1]] = label
    if get_cats and not get_node_labels:
        return node_dict, edge_dict, category
    if get_node_labels and not get_cats:
        return node_dict, edge_dict, node_labels
    if get_node_labels and get_cats:
        return node_dict, edge_dict, node_labels, category
    return node_dict, edge_dict
def pairwise_incidence(H, r):
    """
    Create pairwise adjacency dictionary from hyperedge list dictionary
    Parameters:
        H: a disctionary with edge indices as keys and the nodes they contain as values
        r: maximum hyperedge size
    Returns:
       E: a dictionary with node pairs as keys and the hyperedges they appear in as values
    """
    E = {}
    for e, edge in H.items():
        l = len(edge)
        for i in range(0, l-1):
            for j in range(i + 1, l):
                if (edge[i], edge[j]) not in E:
                    E[(edge[i], edge[j])] = [e]
                else:
                    E[(edge[i], edge[j])].append(e)
        if l < r:
            for node in edge:
                if (node, node) not in E:
                    E[(node, node)] = [e]
                else:
                    E[(node, node)].append(e)
    return E

def get_adjacency(H, n, weighted = True):
    A_dict = {}
    for _, edge in H.items():
        l = len(edge)
        for i in range(l):
            v1 = edge[i]
            for j in range(l):
                v2 = edge[j]
                if v1 != v2:
                    if (v1, v2) not in A_dict:
                        A_dict[(v1, v2)] = 1
                    else:
                        if weighted:
                            A_dict[(v1, v2)] += 1
    first = np.zeros(len(A_dict))
    second = np.zeros(len(A_dict))
    value = np.zeros(len(A_dict))
    for (i, k) in enumerate(A_dict.keys()):
        first[i] = k[0]
        second[i] = k[1]
        value[i] = A_dict[k]
    A = coo_array((value, (first,second)), (n, n))
    return A

def get_largest_connected_component(H, E, n, return_label_map = False):
    A = get_adjacency(H, n)
    G = nx.from_scipy_sparse_array(A)
    G_cc = max(nx.connected_components(G), key = len)
    G_cc = G.subgraph(G_cc).copy()
    nodes = G_cc.nodes()
    node_map = {}
    edge_map = {}
    _H = {}
    _E = {}
    for e, edge in H.items():
        for v in edge:
            if v in nodes:
                _v = get_val(node_map, v)
                _e = get_val(edge_map, e)
                if _v not in _E:
                    _E[_v] =[_e]
                else:
                    _E[_v].append(_e)
                if _e not in _H:
                    _H[_e] = [_v]
                else:
                    _H[_e].append(_v)
    _n = len(_E)
    _r = max([len(edge) for _, edge in _H.items()])
    if return_label_map:
        return _E, _H, _r, _n, node_map
    return _E, _H, _r, _n



