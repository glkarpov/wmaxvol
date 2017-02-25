#! /bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from maxvolpy.maxvol import maxvol
from numpy.polynomial import Chebyshev as T
import numpy.linalg as la
import matplotlib.pyplot as plt
#%matplotlib inline

def dets_of_submatrices(Z):
    y, x = Z.shape
    det = np.zeros([y/2,x/2])
    for i in np.arange(0, y, 2):
        for j in np.arange(0, x, 2):
            temp_mat = [[Z[k][l] for k in range(i, i+2)] for l in range(j, j+2)]
            det[i/2, j/2] = np.linalg.det(temp_mat)  
    det = np.abs(det)        
    ind = np.argwhere(det.max() == det)
    a = 2*ind[0,0]
    b = 2*ind[0,1]
    return (a, b, np.max(np.abs(det)))

def block_maxvol(A, tol=1.0, max_iters=100):
# work on parameters
    if tol < 1:
        tol = 1.0
    N, r = A.shape
    
    #create initial matrix C with I and Z block
    B = np.copy(A, order='F')
    B_inv = np.linalg.inv(B[:r])
    C = np.dot(B, B_inv)
    #print(C)
    #initial indexing
    index = np.arange(N, dtype = np.int32)
      
    # find initial block 2x2 with max determinant in C, if it exists
    i, j, det = dets_of_submatrices(C)
    #print(det, i, j)
    # set number of iters to 0
    iters = 0
    
    while det > tol and iters < max_iters:
        
        index[j] = i
        index[j+1] = i + 1
        
        temp = B[i, :]
        B[i, :] = B[j, :]
        B[j, :] = temp
        
        temp = B[i+1, :]
        B[i+1, :] = B[j+1, :]
        B[j+1, :] = temp
        
        B_inv = np.linalg.inv(B[:r])
        C = np.dot(B, B_inv)
        iters += 1
        i, j, det = dets_of_submatrices(C)
        
    print(index[:r])
    return index[:r]