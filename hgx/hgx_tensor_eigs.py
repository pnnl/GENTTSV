## Tensor times same vector in all but one (TTSV1) and all but two (TTSV2)
from utils import *
from hgx_tensor_algs import *
import numpy as np
from copy import deepcopy
from scipy.sparse.linalg import eigsh
def reduce(E,H,dmax,u,n):
    return ttsv2(E,H,dmax,u,n)
def apply(E,H,dmax,u):
    return ttsv1(E,H,dmax,u)
def LR_evec(A):
    _, v = eigsh(A, k=1, which = "LM", tol = 1e-5, maxiter = 200)
    evec = np.array([_v for _v in v[:, 0]])
    if evec[0] < 0:
        evec = -evec
    return evec / np.linalg.norm(evec, 1)
# Benson and Gleich algorithm for computing the leading Z-eigenvector.
def Z_evec_dynsys(E, H, r, n, tol = 1e-5, niter=200):
    x_init = np.ones(n)/n
    def f(u):
        return LR_evec(reduce(E, H, r, u, n)) - u
    x_curr = deepcopy(x_init)
    h = 0.5
    converged = False
    for i in range(niter):
        print(f'{i} of {niter-1}', flush=True)
        x_next = x_curr + h * f(x_curr)
        s = np.array([a/b for a, b in zip(x_next, x_curr)])
        converged = (np.max(s) - np.min(s)) / np.min(s) < tol
        if converged:
            break
        x_curr = x_next
    evec = x_curr
    return evec, converged
# NQI algorithm for computing the leading H-eigenvector.
def H_evec_NQI(E, H, r, tol = 1e-5, niter=200):
    n = len(E)
    converged = False
    x = np.ones(n)/n
    y = np.abs(np.array(apply(E, H, r, x)))
    for i in range(niter):
        print(f'{i} of {niter-1}', flush=True)
        y_scaled = [_y**(1/(r-1)) for _y in y]
        x = y_scaled / np.linalg.norm(y_scaled, 1)
        y = np.abs(np.array(apply(E, H, r, x)))
        s = [a/(b**(r-1)) for a,b in zip(y, x)]
        converged = (np.max(s) - np.min(s)) / np.min(s) < tol
        if converged:
            break
    return x, converged        
def GCEC(E):
    c = LR_evec(E)
    return c/np.linalg.norm(c, 1), True

def GZEC(E, H, r, n):
    c, converged = Z_evec_dynsys(E, H, r, n)
    return c/np.linalg.norm(c, 1), converged

def GHEC(E, H, r):
    c, converged = H_evec_NQI(E, H, r)
    return c/np.linalg.norm(c, 1), converged

def z_eigen_centrality(hypergraph):
    assert hypergraph.is_connected()
    H, E, r, reverse_node_map = get_dicts_from_hypergraph(hypergraph)
    EE = pairwise_incidence(H, r)
    n = len(E)
    cent, converged = GZEC(EE, H, r, n)
    if not converged:
        print('Iteration did not converge!')
    cent_dict = {}
    for i, _cent in enumerate(cent):
        cent_dict[reverse_node_map[i]] = _cent
    return cent_dict

def h_eigen_centrality(hypergraph):
    assert hypergraph.is_connected()
    H, E, r, reverse_node_map = get_dicts_from_hypergraph(hypergraph)
    cent, converged = GHEC(E, H, r)
    if not converged:
        print('Iteration did not converge!')
    cent_dict = {}
    for i, _cent in enumerate(cent):
        cent_dict[reverse_node_map[i]] = _cent
    return cent_dict
