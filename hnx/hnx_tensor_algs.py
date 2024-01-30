## Tensor times same vector in all but one (TTSV1) and all but two (TTSV2)
from utils import *
import numpy as np
from  numpy import prod
from scipy.special import binom as binomial
from math import factorial
from scipy.sparse import coo_array
from scipy.signal import convolve
def get_dicts_from_hypergraph(h):
    """
    Get edge to nodes dict, node to edges dict, and max. hyperedge size
    Parameters:
        h: hypernetx hypergraph
    Returns:
        H: dict with hyperedges as keys and list of nodes that appear in them as values (dict)
        E: dict with nodes as keys and list of hyperedges they appear in as values (dict)
        r: maximum hyperedge size (int)
        reverse_node_map: dict with old node labels as keys and new node labels as values (dict)
    """
    r = 0
    node_map = {}
    edge_map = {}
    _H = h.incidence_dict
    H = {}
    E = {}
    for _e, _nodes in _H.items():
        e = get_val(edge_map, _e)
        nodes = []
        for _node in _nodes:
            node = get_val(node_map, _node)
            if node not in nodes:
                nodes.append(node)
        H[e] = nodes
        if len(nodes) > r:
            r = len(nodes)
    for edge, nodes in H.items():
        for node in nodes:
            if node not in E:
                E[node] = [edge]
            else:
                E[node].append(edge)
    reverse_node_map = {}
    for old_node_label, new_node_label in node_map.items():
        reverse_node_map[new_node_label] = old_node_label
    return H, E, r, reverse_node_map
        
def pairwise_incidence(H, r):
    """
    Create pairwise adjacency dictionary from hyperedge list dictionary
    Parameters:
        H: a dictionary with edge indices as keys and the nodes they contain as values
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

def COEF(l, r):
    """
    Return the Banerjee alpha coefficient.
    Parameters:
        l: length of given hyperedge
        r: maximum hyperedge size
    Returns:
        alpha: the Banerjee coefficient
    """
    return sum(((-1)**j)*binomial(l, j)*(l - j) ** r for j in range(l+1))

def get_gen_coef_subset_expansion(edge_values, node_value, r):
    k = len(edge_values)
    subset_vector = [0]
    subset_lengths = [0]
    for i in range(k):
        for t in range(len(subset_vector)):
            subset_vector.append(subset_vector[t] + edge_values[i])
            subset_lengths.append(subset_lengths[t] + 1)
    for i in range(len(subset_lengths)):
        subset_lengths[i] = (-1) ** (k - subset_lengths[i])
        # subset_lengths[i] = -1 if (k - subset_lengths[i]) % 2 == 1 else 1
    total = sum([(node_value + subset_vector[i]) ** r * subset_lengths[i] for i in range(len(subset_lengths))])
    return total/factorial(r)
def get_gen_coef_fft(edge_without_node, a, node, l, r):
    coefs = np.array([(a[node]**i)/factorial(i) for i in range(r)])
    for u in edge_without_node:
        _coefs = np.array([a[u]**i/factorial(i) for i in range(r-l+2)])
        _coefs[0] = 0
        coefs = convolve(coefs, _coefs)[0:r]
    gen_fun_coef = coefs[-1]
    return gen_fun_coef
def get_gen_coef_fft_fast_array(edge_without_node, a, node, l, r):
    coefs = [1]
    for i in range(1, r):
        coefs.append(coefs[-1] * a[node] / i)
    coefs = np.array(coefs)
    for u in edge_without_node:
        _coefs = [1]
        for i in range(1, r-l+2):
            _coefs.append(_coefs[-1] * a[u] / i)
        _coefs = np.array(_coefs)
        _coefs[0] = 0
        coefs = convolve(coefs, _coefs)[0:r]
    gen_fun_coef = coefs[-1]
    return gen_fun_coef

def ttsv1(E, H, r, a):
    """
    Compute tensor times same vector in all but one for a hypergraph encoded by E and H using generating functions
    Parameters:
        E: dictionary with nodes as keys and the hyperedges they appear in as values
        H: dictionary with edges as keys and the nodes that appear in them as values
        r: maximum hyperedge size
        a: numpy array encoding the vector by which we wish to multiply the tensor
    Returns:
        s: the tensor times same vector in all but one
    """
    n = len(E)
    s = np.zeros(n)
    r_minus_1_factorial = factorial(r-1)
    for node, edges in E.items():
        c = 0
        for e in edges:
            l = len(H[e])
            alpha = COEF(l, r)
            edge_without_node = [v for v in H[e] if v != node]
            if l == r:
                gen_fun_coef =  prod(a[edge_without_node])
            elif 2 ** (l - 1) < r * (l - 1):
                gen_fun_coef = get_gen_coef_subset_expansion(a[edge_without_node], a[node], r - 1)
            else:
                gen_fun_coef = get_gen_coef_fft_fast_array(edge_without_node, a, node, l, r)
            c += r_minus_1_factorial * l * gen_fun_coef / alpha
        s[node] = c
    return s

def ttsv2(E, H, r, a, n):
    """
    Compute tensor times same vector in all but two for a hypergraph encoded by E and H using generating functions
    Parameters:
        E: dictionary with nodes pairs as keys and the hyperedges they appear in as values
        H: dictionary with edges as keys and the nodes that appear in them as values
        r: maximum hyperedge size
        a: numpy array encoding the vector by which we wish to multiply the tensor
        n: number of nodes
    Returns:
        Y: a sparse array representing the output of TTSV2

    """
    s = {}
    r_minus_2_factorial = factorial(r-2)
    for nodes, edges in E.items():
        v1 = nodes[0]
        v2 = nodes[1]
        c = 0
        for e in edges:
            l = len(H[e])
            alpha = COEF(l, r)
            edge_without_node = [v for v in H[e] if v != v1 and v!= v2]
            if v1 != v2:
                if 2 ** (l - 2) < (r - 2) * (l - 2):
                    gen_fun_coef = get_gen_coef_subset_expansion(a[edge_without_node], a[v1] + a[v2], r - 2)
                else:
                    coefs = [1]
                    for i in range(1, r - 1):
                        coefs.append(coefs[-1] * (a[v1] + a[v2]) / i)
                    coefs = np.array(coefs)
                    for u in H[e]:
                        if u != v1 and u != v2:
                            _coefs = [1]
                            for i in range(1, r-l+2):
                                _coefs.append(_coefs[-1] * a[u] / i)
                            _coefs = np.array(_coefs)
                            _coefs[0] = 0
                            coefs = convolve(coefs, _coefs)[0:r-1]
                    gen_fun_coef = coefs[-1]
            else:
                if 2 ** (l - 1) < (r - 2) * (l - 1):
                    gen_fun_coef = get_gen_coef_subset_expansion(a[edge_without_node], a[v1], r - 2)
                else:
                    coefs = [1]
                    for i in range(1, r - 1):
                        coefs.append(coefs[-1] * (a[v1]) / i)
                    coefs = np.array(coefs)
                    for u in H[e]:
                        if u != v1 and u != v2:
                            _coefs = [1]
                            for i in range(1, r - l + 1):
                                _coefs.append(_coefs[-1] * a[v1] / i)
                            _coefs = np.array(_coefs)
                            _coefs[0] = 0
                            coefs = convolve(coefs, _coefs)[0:r-1]
                    gen_fun_coef = coefs[-1]
            c += r_minus_2_factorial * l * gen_fun_coef / alpha
        s[nodes] = c
        if v1 == v2:
            s[nodes] /= 2
    first = np.zeros(len(s))
    second = np.zeros(len(s))
    value = np.zeros(len(s))
    for (i, k) in enumerate(s.keys()):
        first[i] = k[0]
        second[i] = k[1]
        value[i] = s[k]
    Y = coo_array((value, (first, second)), (n,n))
    return Y + Y.T
