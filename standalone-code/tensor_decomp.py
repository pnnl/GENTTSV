## These tensor functions are based on ones from the paper https://arxiv.org/pdf/1911.03813.pdf by Sherman and Kolda, 2020
from utils import *
from standalone_tensor_algs import *
import numpy as np
from  numpy import prod
from math import factorial
from itertools import combinations_with_replacement
from itertools import product
from copy import deepcopy
from time import time
import random

def objective(x, E, H, q, r, d):
    n = len(E)
    lamb = x[0: q]
    A = x[q:].reshape((n, q))
    Y=np.zeros((n, q))
    for i in range(q):
        a = A[:, i]
        _a = a * d
        Xadm1 = np.array(ttsv1(E, H, r, _a))
        Xadm1 *= d
        Y[:, i]= a ** (r - 1) - Xadm1
    w = np.zeros(q)
    for i in range(q):
        w[i]=A[:, i] @ Y[:, i]
    B = A.T @ A
    C = B ** (r - 1)
    u = (C * B) @ lamb
    f = lamb  @ u - 2 * w @ lamb
    return f

def objective_unnormed(x, E, H, q, r, d):
    n = len(E)
    lamb = x[0: q]
    A = x[q:].reshape((n, q))
    Y=np.zeros((n, q))
    for i in range(q):
        a = A[:, i]
        _a = a * d
        Xadm1 = np.array(ttsv1(E, H, r, _a))
        Xadm1 *= d
        Y[:, i]= a ** (r - 1) - Xadm1
    w = np.zeros(q)
    for i in range(q):
        w[i]=A[:, i] @ Y[:, i]
    B = A.T @ A
    C = B ** (r - 1)
    u = (C * B) @ lamb
    f = lamb  @ u - 2 * w @ lamb
    return f

def fg_implicit(x, E, H, q, r, d):
    n = len(E)
    lamb = x[0: q]
    A = x[q:].reshape((n, q))
    Y=np.zeros((n, q))
    for i in range(q):
        a = A[:, i]
        _a = a * d
        Xadm1 = np.array(ttsv1(E, H, r, _a))
        Xadm1 *= d
        Y[:, i]= a ** (r-1) - Xadm1
    w = np.zeros(q)
    for i in range(q):
        w[i]=A[:, i] @ Y[:, i]
    B = A.T @ A
    C = B ** (r - 1)
    u = (C * B) @ lamb
    f = lamb  @ u - 2 * w @ lamb
    g_lamb = -2 * (w - u)
    for i in range(q):
        A[:, i] *= lamb[i]
    G_A = -2 * r * (Y - A @ C)
    for i in range(q):
        G_A[: ,i] *= lamb[i]
    grad = np.zeros(q * (n + 1))
    grad[0: q] = g_lamb
    grad[q:] = G_A.flatten(order = 'F')
    return (f, grad)

def gradient(x, E, H, q, r, d):
    n = len(E)
    lamb = x[0: q]
    A = x[q:].reshape((n, q))
    Y=np.zeros((n, q))
    for i in range(q):
        a = A[:, i]
        _a = a * d
        Xadm1 = np.array(ttsv1(E, H, r, _a))
        Xadm1 *= d
        Y[:, i]= a ** (r-1) - Xadm1
    w = np.zeros(q)
    for i in range(q):
        w[i]=A[:, i] @ Y[:, i]
    B = A.T @ A
    C = B ** (r - 1)
    u = (C * B) @ lamb
    g_lamb = -2 * (w - u)
    for i in range(q):
        A[:, i] *= lamb[i]
    G_A = -2 * r * (Y - A @ C)
    for i in range(q):
        G_A[: ,i] *= lamb[i]
    grad = np.zeros(q * (n + 1))
    grad[0: q] = g_lamb
    grad[q:] = G_A.flatten(order = 'F')
    return grad


def hypergraph_pruner(h, r, deg_frac):
    import copy
    
    _H = copy.deepcopy(h)
    H = {}
    _node_map = {}
    _edge_map = {}
    
    for e, nodes in copy.deepcopy(_H).items():
        if len(nodes) <= r:
            for node in nodes:
                _e = get_val(_edge_map, e)
                _node = get_val(_node_map, node)
                if _e not in H.keys():
                    H[_e] = [_node]
                else:
                    H[_e].append(_node)
    
    E = {}
    for e, nodes in H.items():
        for node in nodes:
            if node not in E.keys():
                E[node] = [e]
            else:
                E[node].append(e)
    
    m = len(H)
    node_map = {}
    edge_map = {}
    remove_dict = {}
    
    for v, edges in E.items():
        if len(edges) / m > deg_frac:
            remove_dict[v] = 1
    
    _H = copy.deepcopy(H)
    
    for e, nodes in _H.items():
        for node in nodes:
            if node in remove_dict.keys():
                H[e].remove(node)
        if len(H[e]) < 2:
            del H[e]
    
    _H = {}
    _E = {}
    
    for e in H.keys():
        _e = get_val(edge_map, e)
        
        for node in H[e]:
            _node = get_val(node_map, node)
            if _e not in _H.keys():
                _H[_e] = [_node]
            else:
                _H[_e].append(_node)
            
            if _node not in _E.keys():
                _E[_node] = [_e]
            else:
                _E[_node].append(_e)
    
    node_map_ = {}
    for v in _node_map.keys():
        if _node_map[v] in node_map.keys():
            node_map_[v] = node_map[_node_map[v]]
    
    return _E, _H, node_map_

def get_parsed_hyper(name, r, deg_frac):
    _E, _H, _cats = parse_hyper(name, min_edge_size = 2, max_edge_size = r, get_cats = True)
    E, H, node_map = hypergraph_pruner(_H, r, deg_frac)
    inv_node_map = {v:k for k,v in node_map.items()}
    category = np.zeros(len(node_map))
    for node in E:
        category[node] = _cats[inv_node_map[node]]
    return E, H, category
