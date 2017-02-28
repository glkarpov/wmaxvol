#! /bin/env python
# -*- coding: utf-8 -*-

# import ahkab
# from ahkab import circuit, printing, time_functions, dc_analysis, utilities, dc_guess, devices, ekv, transient
import numpy as np
import matplotlib.pyplot as plt
import scipy
from numpy.polynomial.hermite import hermval
from numpy.polynomial.chebyshev import chebval, chebder
from numpy.polynomial import Chebyshev as T
from scipy.special.orthogonal import h_roots
import itertools
import rect_maxvol
# from scipy.linalg import solve_triangular, get_lapack_funcs, get_blas_funcs
# import sympy
# from sympy import *
# %matplotlib inline


# Combinatorials funcs

norm_cheb = 1.0/np.sqrt(np.pi/2.0)

def binom_sh(p,l):
    """
    Shifted binomial:
    (p+l\\p) = (p+l)!/p!*l!
    meaning number of monoms to approx. function, with l vars and poly. power <= p
    """
    return np.math.factorial(p+l)/(np.math.factorial(p)*np.math.factorial(l))

def indeces_K(l, p):
    """
    returns all vectors of length l with sum of indices <= p, starting form 0
    """
    for cmb_u in itertools.combinations_with_replacement(xrange(p+1), l):
        for cmb in set(itertools.permutations(cmb_u)):
            if sum(cmb) <= p:
                yield cmb

def indeces_K_cut(l, maxn):
    """
    MAGIC FUNCTION
    p is determines automatically
    """
    p = int(float(  (maxn*np.math.factorial(l))**(1.0/float(l))  )+1)
    while binom_sh(p, l) < maxn:
        print('THIS NEVER HAPPENS!!!\n')
        p += 1
    a = list(indeces_K(l, p))
    a = sorted(a, key=lambda e: max(e))
    a = sorted(a, key=lambda e: sum(e))
    return a[:maxn]



# Some with polynomials

def herm_mult_many(x, xi, poly_func=None):
    """
    INPUT
    x - array of point where to calculate (np.array N x l)
    xi - array of powers of Hermite (or non-Hermite) poly (array of length l)
    
    OUTPUT
    [H_xi[0](x[0, 0])*H_xi[1](x[0, 1])*...,
     H_xi[0](x[1, 0])*H_xi[1](x[1, 1])*...,
                    ...
     H_xi[0](x[N-1, 0])*H_xi[1](x[N-1, 1])*...,]
    """
    N, l = x.shape
    assert(l == len(xi))

    if poly_func is None:
        poly_func = [herm] * l

    res = np.ones(N)
    for n in xrange(l):
        res *= poly_func[n](x[:, n], xi[n])

    return res


def herm_mult_many_diff(x, xi, diff_var, poly_func=None, poly_diff=None):
    """
    INPUT
    x - array of point where to calculate (np.array N x l)
    xi - array of powers of Hermite poly (array of length l)
    
    OUTPUT
    [H_xi[0](x[0, 0])*H_xi[1](x[0, 1])*...,
     H_xi[0](x[1, 0])*H_xi[1](x[1, 1])*...,
                    ...
     H_xi[0](x[N-1, 0])*H_xi[1](x[N-1, 1])*...,]
    """
    N, l = x.shape
    assert(l == len(xi))

    if poly_func is None:
        poly_func = [herm] * l
        poly_diff = [herm_diff] * l

    res = np.ones(N)
    for n in xrange(l):
        if n == diff_var:
            res *= poly_diff[n](x[:, n], xi[n])
        else:
            res *= poly_func[n](x[:, n], xi[n])
    

    return res

# Some orthogonal polynomials

def cheb(x, n):
    """
    returns T_n(x)
    value of not normalized Chebyshev polynomial
    $\int \frac1{\sqrt{1-x^2}}T_m(x)T_n(x) dx = \delta_{nm}$
    """
    return T.basis(n)(x)

def cheb_diff(x, n):
    return T.basis(n).deriv(1)(x)



def herm(x, n):
    """
    returns H_n(x)
    value of normalized Probabilistic polynomials
    $\int exp(-x^2/2)H_m(x)H_n(x) dx = \delta_{nm}$
    """
    cf = np.zeros(n+1)
    cf[n] = 1
    #return hermval(x, cf)
    nc = ((2.0*np.pi)**(0.25)) * np.sqrt(float(np.math.factorial(n)))
    return (2**(-float(n)*0.5))*hermval(x/np.sqrt(2.0), cf)/nc


def herm_diff(x, n):
    return x*herm(x, n-1) if n>0 else 0

# Main func

def GenMat(n_size, x, poly=None, poly_diff=None, debug=False):
    """
    INPUT
        n_size — number of colomns (monoms), int
        x — points, n2 x l numpy array (n2 is arbitrary integer, number of point, l — number of independent vars = number of derivatives  )
    OUTPUT 
        n2*(l+1) x n_size matrix A, such that 
        a_{ij} = H_i(x_j) when i<l 
        or a_{ij}=H'_{i mod l}(x_j), where derivatives are taken on coordinate with number i//l
    """

    n2, l = x.shape
    nA = n2*(l+1) # all values in all points plus all values of all derivatives in all point: n2 + n2*l
    A = np.zeros((nA, n_size))
    if debug:
        print('number of vars(n2) = {}, dim of space (number of derivatives, l) = {},  number of monoms(n_size) = {}'.format(n2, l, n_size))

    for i, xp in enumerate(indeces_K_cut(l, n_size)):
        if debug:
            print ('monom #{} is {}'.format(i, xp))
        A[0:n2, i] = herm_mult_many(x, xp, poly)
        for dl in xrange(1, l+1):
            A[n2*dl:n2*dl+n2, i] = herm_mult_many_diff(x, xp, dl-1, poly, poly_diff)

    return A

def CronProdX(wts, rng):
    """
    INPUT 
        wts -- wight functions (np.array  n x l) (typically, wts[:, 0] == wts[:, 1] == ... if poly are the same)
        rng -- range of variables (matrix of n x l dim.)
    """

    n, l = rng.shape
    n2 = n**l
    x = np.zeros((n2, l))
    wa = np.zeros((n2, l))
    w = np.zeros((n2,))
    for xn, perm in enumerate(itertools.product(xrange(n), repeat=l)):
        #print xn, perm
        x[xn , :]  = [rng[val, i] for i, val in enumerate(perm)]
        wa[xn, :] = [wts[val, i] for i, val in enumerate(perm)]
        w[xn] = (np.prod(wa[xn, :]))**(1.0/l)

    return x, w

def RenormXAndIdx(res, x):
    cf = x.shape[0]
    exists_idx = np.array(list(set( res % cf  )))
    resnew = np.zeros( (len(res), 2 ))
    for idx, i in enumerate(res):
        n, pos = divmod(i, cf)
        resnew[idx, 0] = np.where(exists_idx==pos)[0][0]
        resnew[idx, 1] = n
        
    return resnew, x[exists_idx, :]

def PlotPoints(res, xout):
    plt.clf()
    plt.hold()
    plt.scatter(xout[:,0], xout[:,1], facecolors='None', s=20)
    # plt.hold()

    for pos, dl in res:
        if dl == 0:
            color = 'r'
            ss_circ = 500
        if dl == 1:
            color = 'b'
            ss_circ = 150
        if dl == 2:
            color = 'g'
            ss_circ = 60
        assert(dl < 3)
        plt.hold()
        plt.scatter(xout[pos, 0], xout[pos, 1], facecolors=color, s=ss_circ, alpha=0.3, edgecolors='face')
        plt.hold()
    plt.savefig('points.pdf', bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    print ('Test run')

    num_p = 4 # number of points we select from on each axis.
    l = 2
    #Number of rows in the matrix will be (num_p+1)**l*(l+1)
    A_size = 75 # number of columns in matrix (numb. of monoms)
    x, w = h_roots(num_p + 1)
    x_in = np.array([list(x)]*l).T
    w_in = np.array([list(w)]*l).T
    x_many, w_many  = CronProdX(w_in, x_in)
    A = GenMat(A_size, x_many)
    print (A.shape)
    print (A)
    print ("Rank =", np.linalg.matrix_rank(A))

    # Random x's
    x_many = np.random.rand((num_p+1)**l, l)
    A = GenMat(A_size, x_many)
    print (A.shape)
    print ("Rank (random matrx.) =", np.linalg.matrix_rank(A))

    # MAXVOL!!!
    # New, big matrix!

    num_p = 8 # number of points we select from on each axis.
    l = 2
    A_size = 20 # number of columns in matrix (numb. of monoms)
    x, w = h_roots(num_p + 1)
    x_in = np.array([list(x)]*l).T
    w_in = np.array([list(w)]*l).T
    x_many, w_many  = CronProdX(w_in, x_in)
    A = GenMat(A_size, x_many)
    print ("Rank (maxvol matrx) =", np.linalg.matrix_rank(A))

    n2 = A.shape[0]/(l+1)
    for i in xrange(A.shape[0]):
        A[i, :] *= w_many[i % n2]
    res, _ = rect_maxvol.rect_maxvol(A, minK=A_size, maxK=A_size)
    A = A[res, :]

    # remove unnecessary x
    res, x = RenormXAndIdx(res, x_many)
    PlotPoints(res, x)


