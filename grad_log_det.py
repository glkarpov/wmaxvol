
# coding: utf-8

# import numpy as np
import autograd.numpy as np
import autograd.numpy.linalg as LA
from numpy.polynomial import Chebyshev as T
import gen_mat as gen
from numba import jit

"""
These two functions are accelerated (precompilled and parallelized) by using @jit decorator from numba 
"""

# it is needed to know number of colums (num_col) in model matrix A and dimension (dim) of the model
global num_col  
global dim  

@jit(parallel = True, nogil = True)
def grad(points):
    """
    Returns analytically calculated gradient of objective function: 
    
    grad(-log(det(A.T*A))).
    
    A is a tall matrix of the model:
    
    Row – n-dimensional point,
    Column – value of expansion term (e.g. T_0(x_1)*T_1(y_1), n = 2) in corresponding point. 
    
    !! Basis polynomials – Chebyshev !!
    
    INPUT
        points – 1-D vector of points
    OUTPUT
        grad – gradient of objective function in given points
    """
    if (len(points.shape) == 1): #transform 1-D vector of points to the matrix (len(points)/n) x (n)
        points = np.stack(np.split(points, dim),1)
        
    idx = gen.indeces_K_cut(dim, num_col) 
    max_degree = np.max(idx)
    
    num_of_points = points.shape[0]
    
    # computing values of all possible Chebyshev polinomials (and its derivatives) in the input points 
    T_deriv = np.zeros((num_of_points*dim, max_degree + 1), dtype = np.float64) 
    T_val = np.zeros((num_of_points*dim, max_degree + 1), dtype = np.float64) 
    for i in range(max_degree + 1):
        T_deriv[:, i] = T.deriv(T.basis(i))(points.T.ravel()[:])
        T_val[:, i] = T.basis(i)(points.T.ravel()[:])
    
    # key part of analytical calculation
    # here is implemented analytical formula (for multidimensional case)
    grad = np.zeros(num_of_points*dim, dtype = np.float64)    
    A = np.split(gen.GenMat(num_col, points, poly = gen.cheb, poly_diff = gen.cheb_diff), dim + 1)[0]
    B_inv = LA.inv(np.dot(A.conj().T, A))
    for k in range(len(grad)):
        for i in range(B_inv.shape[0]):
            for j in range(B_inv.shape[0]):
                col = k//num_of_points
                row = k%num_of_points
                grad[k] += B_inv[j,i] * (A[row,i]*A[row,j]*(T_deriv[k,idx[i][col]]/T_val[k,idx[i][col]] + T_deriv[k,idx[j][col]]/T_val[k,idx[j][col]]))
    return -grad

@jit(parallel = True, nogil = True)
# this is stable calculation of objective function to minimize -log(det(A.T*A)) 
def loss_func(points):
    points = points.reshape(points.size // dim, dim, order='F')
    A = gen.GenMat(num_col, points, poly = gen.cheb, poly_diff = gen.cheb_diff, ToGenDiff=False)
    S = LA.svd(A, compute_uv = False)
    ld = 2.0*np.sum(np.log(S))
    return -ld
