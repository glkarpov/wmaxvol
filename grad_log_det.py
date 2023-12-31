
# coding: utf-8
from __future__ import print_function

import numpy as np
import numpy.linalg as LA
# import autograd.numpy as np
# import autograd.numpy.linalg as LA

from numpy.polynomial import Chebyshev as T
import gen_mat as gen
from numba import jit

"""
These two functions are accelerated (precompilled and parallelized) by using @jit decorator from numba 
"""

# it is needed to know number of colums (num_col) in model matrix A and dimension (dim) of the model
global num_col
global dim
tol = 1e-12

@jit(parallel = True, nogil = True)
def grad(points, poly=gen.cheb):
    """
    Returns analytically calculated gradient of objective function: 

    grad(-log(det(A.T*A))).

    A is a tall matrix of the model:

    Row – n-dimensional point,
    Column – value of expansion term (e.g. T_0(x_1)*T_1(y_1), n = 2) in corresponding point. 

    !! Basis polynomials – Chebyshev !!

    INPUT
        points – 1-D vector of points, Fortran order of points components: first are going 0 components of each point, then 1 and so on
    OUTPUT
        grad – gradient of objective function in given points
    """

    if (points.ndim == 1 or points.shape[1] == 1): #transform 1-D vector of points to the matrix (len(points)/n) x (n)
        points = points.reshape(-1, dim, order='F')

    idx = np.array(gen.indeces_K_cut(dim, num_col))
    max_degree = np.max(idx)

    num_of_points = points.shape[0]
    tot_elems = points.size

    # computing values of all possible Chebyshev polynomials (and its derivatives) in the input points 
    T_deriv = np.empty((tot_elems, max_degree + 1), dtype = points.dtype)
    T_val   = np.empty((tot_elems, max_degree + 1), dtype = points.dtype)
    points_flat = points.ravel('F')
    for i in range(max_degree + 1):
        T_deriv[:, i] = poly.diff(points_flat, i)
        T_val[:, i]   = poly     (points_flat, i)

    A = gen.GenMat(num_col, points, poly_vals=T_val, indeces=idx,  ToGenDiff=False)

    _, S, V = LA.svd(A, full_matrices = False)
    S[ S < tol ] = tol
    B_inv = V.T.dot(np.diag(1./(S**2))).dot(V)

    # key part of analytical calculation
    # here is implemented analytical formula (for multidimensional case)

    grad_vec = np.zeros(tot_elems, dtype = points.dtype)
    for k in range(tot_elems):
        col = k//num_of_points
        row = k%num_of_points
        A_row = A[row]
        idx_col = idx.T[col]
        for i in range(B_inv.shape[0]):
            alpha = T_deriv[k, idx_col[i]]/T_val[k, idx_col[i]]
            for j in range(B_inv.shape[0]):
                grad_vec[k] += B_inv[j,i] * (A_row[i]*A_row[j]*(alpha  + T_deriv[k, idx_col[j]]/T_val[k, idx_col[j]]))
    return -grad_vec

@jit(parallel = True, nogil = True)
# this is stable calculation of objective function to minimize -log(det(A.T*A)) 
def loss_func(points, poly=gen.cheb, ToGenDiff=False):
    if (points.ndim == 1 or points.shape[1] == 1):
        points = points.reshape(-1, dim, order='F')
    A = gen.GenMat(num_col, points, poly=poly, ToGenDiff=ToGenDiff)
    S = LA.svd(A, compute_uv = False)
    S[ S < tol ] = tol
    ld = 2.0*np.sum(np.log(S))
    return -ld






"""# Test part -------------------------------------------------------------------------------------------------------
if __name__ == '__main__':

    num_col = 5
    num_points = 7
    dim = 2

    l_bound = -3.
    u_bound = 3.

    np.random.seed(42)
    x_0 = l_bound + (u_bound - l_bound)*np.random.rand(num_points*dim)

    gradF = grad(x_0)
    print(gradF)
"""
