#! /bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import scipy

def maxvol_swm(A, tol=1.05, max_iters=100, top_k_index=-1):

    # work on parameters
    if tol < 1:
        tol = 1.0
    N, r = A.shape
    if N <= r:
        return np.arange(N, dtype=np.int32), np.eye(N, dtype=A.dtype)
    if top_k_index == -1 or top_k_index > N:
        top_k_index = N
    if top_k_index < r:
        top_k_index = r
    
    #create matrix C with I and Z block
    B = np.copy(A[:top_k_index], order='F')
    B_inv = np.linalg.inv(B[:r])
    C = np.dot(B, B_inv)
    
    #initial indexing
    index = np.arange(N, dtype=np.int32)
      
    # find initial max value in C, if it exists
    i, j = divmod(abs(C[:,:]).argmax(), r)
        
    # set number of iters to 0
    iters = 0
      
    # check if need to swap rows
    while abs(C[i,j]) > tol and iters < max_iters:
        # add i to index and recompute C by SWM-formula
        index[j] = i
        
        tmp_row = C[i, :].copy()
        tmp_row[j] -= 1.
        
        tmp_column = C[:,j].copy()
        tmp_column[j] -= 1.
        tmp_column[i] += 1.
        
        tmp_column = np.matrix(tmp_column).T
        tmp_row = np.matrix(tmp_row)
        
        alpha = -1./C[i,j]
        C += alpha*tmp_column.dot(tmp_row)
        
        iters += 1
        i, j = divmod(abs(C[:,:]).argmax(), r)
    return index[:r]
