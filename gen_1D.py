#! /bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import scipy
from numpy.polynomial.chebyshev import chebval, chebder
from numpy.polynomial import Chebyshev as T

def cheb(x, n):
    """
    returns T_n(x)
    value of not normalized Chebyshev polynomial
    $\int \frac1{\sqrt{1-x^2}}T_m(x)T_n(x) dx = \delta_{nm}$
    """
    return T.basis(n)(x)

def cheb_diff(x, n):
    return T.basis(n).deriv(1)(x)

def Gen_1D(n_size, x, debug=False):
    l = x.shape[0]
    n2 = l * 2
    A = np.zeros((n2, n_size), dtype=float)
    if debug:
        print('number of points(l) = {},  number of monoms(n_size) = {}'.format(l, n_size))
    
    for i in range(n_size):
        A[0:l, i] = cheb(x[:] , i)
        A[l:n2, i] = cheb_diff(x[:], i)
    
    return (A)

def Gen_1D_coupled(n_size, x, debug=False):
    l = x.shape[0]
    n2 = l * 2
    A = np.zeros((n2, n_size))
    if debug:
        print('number of points(l) = {},  number of monoms(n_size) = {}'.format(l, n_size))
        
    for i in range(n_size):
        A[0:n2 - 1:2, i] = cheb(x[:] , i)
        A[1:n2:2, i] = cheb_diff(x[:], i)
        
    return (A)    