#! /bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
import numpy as rnp
import autograd.numpy as np
import matplotlib.pyplot as plt
import scipy
from numpy.polynomial.hermite import hermval
from numpy.polynomial import Chebyshev as T
# from numpy.polynomial import Hermite as H
from numpy.polynomial import Legendre as L
import itertools
from numba import jit
# from scipy.linalg import solve_triangular, get_lapack_funcs, get_blas_funcs
# import sympy
# from sympy import *
from copy import copy, deepcopy


xrange = range


# Combinatorials funcs

# norm_cheb = 1.0/np.sqrt(np.pi/2.0)
sqrt_pi = np.sqrt(np.pi)
sqrt_pi2 = np.sqrt(np.pi*2)

@jit
def ReverseIdx(idx):
    """
    returns Reverse permutation
    -1 on unknown places
    if idx[i] = f
    then res[f] = i
    """
    n2 = max(idx) + 1
    NumToIdxInv = np.full(n2, -1, dtype=int)
    for ni, i in enumerate(idx):
        NumToIdxInv[i] = ni

    return NumToIdxInv

@jit
def sort_like(ar, arn):
    """
    RETURNS idx so that
    ar == arn[idx]
    """
    return np.argsort(ReverseIdx(ar)[arn])


# @jit
# def change_intersept(inew, iold):
    # """
    # change two sets of rows or columns when indices may intercept with preserving order
    # RETURN two sets of indices,
    # than say A[idx_n] = A[idx_o]
    # """
    # # union = np.array(list( set(inew) | set(iold) ))
    # union = np.union1d(inew, iold)
    # idx_n = np.hstack((inew, np.setdiff1d(union, inew)))
    # idx_o = np.hstack((iold, np.setdiff1d(union, iold)))
    # return  idx_n, idx_o


@jit('i8(i8,i8)')
def binom_sh(p,l):
    """
    Shifted binomial:
    (p+l\\p) = (p+l)!/p!*l!
    meaning number of monoms to approx. function, with l vars and poly. power <= p
    """
    return int(  rnp.math.factorial(p+l)//(rnp.math.factorial(p)*rnp.math.factorial(l))  )

def OnesFixed(m, n):
    """
    m ones on n places
    """
    for i in itertools.combinations_with_replacement(xrange(n), m):
        uniq = np.unique(i)
        if len(uniq) == len(i):
            res = np.full(n, False)
            res[uniq] = True
            yield res

@jit
def indeces_K(l, q, p=1):
    """
    returns all vectors of length l with sum of indices in power p <= q^p, starting form 0
    x^p + y^p <= q^p
    Elements can repeat!
    """
    qp = q**p
    m = int(qp) # max number of non-zero elements
    if m >= l:
        # for cmb_u in itertools.combinations_with_replacement(xrange(q+1), l):
            # for cmb in set(itertools.permutations(cmb_u)):
        for cmb in itertools.product(xrange(q+1), repeat=l):
            if sum(np.array(cmb)**p) <= qp:
                yield cmb
    else:
        ones = list(OnesFixed(m, l))
        for cmb in itertools.product(xrange(q+1), repeat=m): # now m repeat
            if sum(np.array(cmb)**p) <= qp:
                for mask in ones:
                    res = np.zeros(l, dtype=int)
                    res[mask] = cmb
                    yield tuple(res)


def indeces_K_cut(l, maxn, p=1, q=1):
    """
    MAGIC FUNCTION
    q is determined automatically
    """
    # q = int(float(  (maxn*np.math.factorial(l))**(1.0/float(l))  )+1)
    while binom_sh(q, l) < maxn:
        # print('THIS NEVER HAPPENS!!!\n')
        q += 1
    # a = list(set(  (tuple(i) for i in  indeces_K(l, q, p)   ) ))
    a = indeces_K(l, q, p)
    # max_pow = long(max(max(i) for i in a))
    # a = sorted(a, key=lambda e: ''.join(str(i) for i in e), reverse=True)
    # a = sorted(a, key=lambda e: sum([ ((1L+max_pow)**ni)*i for ni, i in enumerate(e) ]), reverse=True)
    a = sorted(a, reverse=True)
    a = [el for el, _ in itertools.groupby(a)] # delete duplicates
    a = sorted(a, key=lambda e: max(e))
    a = sorted(a, key=lambda e: np.sum( np.array(e)**p ))
    if len(a) < maxn:
        return indeces_K_cut(l, maxn, p, q+1)
    else:
        return a[:maxn]


def num_of_indeces_K(l, q, max_p):
    a = indeces_K(l, q, max_p)
    a = [el for el, _ in itertools.groupby(a)] # delete duplicates
    return len(a)



# Some with polynomials
@jit
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

    # res = np.ones(N)
    # for n in xrange(l):
    #    res *= poly_func[n](x[:, n], xi[n])
    res = poly_func[0](x[:, 0], xi[0])
    for n in xrange(1, l):
        res *= poly_func[n](x[:, n], xi[n])

    return res


@jit
def herm_mult_many_diff(x, xi, diff_var, poly_func=None):
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

    res = np.ones(N, dtype=x.dtype)
    for n in xrange(l):
        if n == diff_var:
            res *= poly_func[n].diff(x[:, n], xi[n])
        else:
            res *= poly_func[n](x[:, n], xi[n])
    

    return res

# Some orthogonal polynomials

def cheb(x, n):
    """
    returns T_n(x)
    value of not normalized Chebyshev polynomial
    $\int \frac1{\sqrt{1-x^2}}T_m(x)T_n(x) dx = \frac\pi2\delta_{nm}$
    """
    return T.basis(n)(x)

def cheb_diff(x, n):
    return T.basis(n).deriv(1)(x)



def cheb_snorm(n):
    return np.pi/2.0 if n != 0 else np.pi

cheb.diff = cheb_diff
cheb.snorm = cheb_snorm

def herm_nn(x, n):
    """
    returns H_n(x)
    value of non-normalized Probabilistic polynomials
    $\int exp(-x^2/2)H_m(x)H_n(x) dx = \delta_{nm}$
    """
    cf = np.zeros(n+1)
    cf[n] = 1
    return (2**(-float(n)*0.5))*hermval(x/np.sqrt(2.0), cf)

def herm_diff_nn(x, n):
    if n <= 0:
        return x*0.0

    cf = np.zeros(n)
    cf[n-1] = 1
    return 2**(0.5*(1.0-float(n)))*n*hermval(x/np.sqrt(2.0), cf)

def herm_snorm(n):
    """
    Square norm of "math" Hermite (exp(-x^2/2))
    """
    return rnp.math.factorial(n)


herm_nn.diff = herm_diff_nn
herm_nn.snorm = herm_snorm


@jit
def herm(x, n):
    """
    returns H_n(x)
    value of normalized Probabilistic polynomials
    $\int exp(-x^2/2)H_m(x)H_n(x) dx = \delta_{nm}$
    """
    cf = np.zeros(n+1)
    cf[n] = 1
    nc = np.sqrt(float(rnp.math.factorial(n))) # norm
    return (2**(-float(n)*0.5))*hermval(x/np.sqrt(2.0), cf)/nc

# @jit
def herm_diff(x, n):
    if n <= 0:
        return x*0.0

    cf = np.zeros(n)
    cf[n-1] = 1
    nc = np.sqrt(float(rnp.math.factorial(n))) # norm
    return 2**(0.5*(1.0-float(n)))*n*hermval(x/np.sqrt(2.0), cf)/nc

def herm_norm_snorm(n):
    """
    For uniform
    """
    return 1.0

herm.diff = herm_diff
herm.snorm = herm_norm_snorm

# @jit
def trigpoly(xin, n, interval=(-1,1)):
    """
    return sin(n x) or cos(n x)
    """
    if n==0:
        try:
            return np.full(xin.shape, 1.0)
        except:
            # xin is number, not array
            return 1.0

    x = np.pi*(interval[0] + interval[1] - 2.0*xin)/(interval[0] - interval[1]) # map interval to [-pi, pi]
    func = np.cos if n % 2 else np.sin
    tpow = (n+1) // 2

    return func(tpow*x)

# @jit
def trigpoly_diff(xin, n, interval=(-1,1)):
    if n==0:
        return x*0.0

    x = np.pi*(interval[0] + interval[1] - 2.0*xin)/(interval[0] - interval[1]) # map interval to [-pi, pi]
    func = (lambda x: -np.sin(x)) if n % 2 else np.cos
    tpow = (n+1) // 2

    return tpow*func(tpow*x)

trigpoly.diff = trigpoly_diff


@jit
def legendre(x, n, interval=(-1.0, 1.0)):
    """
    Non-normed poly
    """
    xn = (interval[0] + interval[1] - 2.0*x)/(interval[0] - interval[1])
    return L.basis(n)(xn)

@jit
def legendre_diff(x, n, interval=(-1.0, 1.0)):
    xn = (interval[0] + interval[1] - 2.0*x)/(interval[0] - interval[1])
    return L.basis(n).deriv(1)(xn)

def legendre_snorm(n, interval=(-1.0, 1.0)):
    """
    RETURNS E[L_n L_n]
    """
    # return 2.0/(2.0*n + 1.0)
    return (interval[1] - interval[0])/(2.0*n + 1.0)

legendre.diff = legendre_diff
legendre.snorm = legendre_snorm


def MakeNormPoly(poly):
    def f(x, n, **kwargs):
        return poly(x, n, **kwargs)/np.sqrt( poly.snorm(n, **kwargs) )

    def fdiff(x, n, **kwargs):
        return poly.diff(x, n, **kwargs)/np.sqrt( poly.snorm(n, **kwargs) )

    f.__name__ = poly.__name__ + '_normed'
    f.diff = fdiff
    f.snorm = lambda n : 1.0
    return f

legendre.normed = MakeNormPoly(legendre)


# Main func

@jit
def GenMat(n_size, x, poly=None, poly_diff=None, debug=False, pow_p=1, indeces=None, ToGenDiff=True, IsTypeGood=True, poly_vals=None, poly_diff_vals=None):
    """
    INPUT
        n_size — number of colomns (monoms), int
        x — points, num_pnts x l numpy array (num_pnts is arbitrary integer, number of point, l — number of independent vars = number of derivatives  )
    OUTPUT 
        num_pnts*(l+1) x n_size matrix A, such that 
        a_{ij} = H_i(x_j) when i<l 
        or a_{ij}=H'_{i mod l}(x_j), where derivatives are taken on coordinate with number i//l
    """

    num_pnts, l = x.shape

    ss = """<class 'autograd"""
    IsTypeGood = IsTypeGood and str(x.__class__)[:len(ss)] != ss


    calc_local_vals = False
    if poly is not None:
        use_func = True
        if not isinstance(poly, list):
            assert callable(poly), "poly must be either a func or a list of funcs"
            if IsTypeGood:
                # 'cause np.copy and deepcopy cannot handle autograd object
                calc_local_vals = True
            else:
                # if not calc_local_vals:
                poly = [poly] * l
    else:
        assert poly_vals is not None, "Neither poly nor poly_vals parameter got"
        use_func = False
        if poly_diff_vals is None:
            ToGenDiff = False

    if poly_diff is not None:
        print ("""Parameter "poly_diff" is obsolete! Do not use it in func 'GenMat'""")

    """
    if poly_diff is not None:
        if not isinstance(poly_diff, list):
            assert(callable(poly_diff))
            poly_diff = [poly_diff] * l
    """

    if indeces is None:
        indeces = indeces_K_cut(l, n_size, p=pow_p)
    else:
        assert(len(indeces) == n_size)


    if ToGenDiff:
        nA = num_pnts*(l+1) # all values in all points plus all values of all derivatives in all point: num_pnts + num_pnts*l
    else:
        nA = num_pnts


    if IsTypeGood:
        A = np.empty((nA, n_size), dtype=x.dtype)
    else:
        A = []

    # assert IsTypeGood or not ToGenDiff or not use_func, 'Not implemented yet'


    if calc_local_vals:
        tot_elems = x.size
        max_degree = np.max(indeces)
        # poly_vals = np.empty((tot_elems, max_degree + 1), dtype = x.dtype)
        poly_vals = []
        for i in range(max_degree + 1):
            poly_vals.append( poly(x.ravel('F'), i ) )
        poly_vals = np.vstack(poly_vals).T
        # print (type(poly_vals))

        if ToGenDiff:
            poly_diff_vals = []
            for i in range(max_degree + 1):
                poly_diff_vals.append( poly.diff(x.ravel('F'), i ) )
            poly_diff_vals = np.vstack(poly_diff_vals).T

        use_func = False


    if debug:
        print('number of vars(num_pnts) = {}, dim of space (number of derivatives, l) = {},  number of monoms(n_size) = {}'.format(num_pnts, l, n_size))


    if use_func: # call poly
        for i, xp in enumerate(indeces):
            # if debug:
                # print ('monom #{} is {}'.format(i, xp))
            Acol = []
            if IsTypeGood:
                A[:num_pnts, i] = herm_mult_many(x, xp, poly)
            else:
                Acol.append(herm_mult_many(x, xp, poly))

            if ToGenDiff:
                if IsTypeGood:
                    for dl in xrange(1, l+1):
                        A[num_pnts*dl:num_pnts*dl+num_pnts, i] = herm_mult_many_diff(x, xp, dl-1, poly)
                else:
                    for dl in xrange(1, l+1):
                        Acol.append(herm_mult_many_diff(x, xp, dl-1, poly))

                A.append(np.hstack(Acol))
            else:
                A.append(Acol[0])


    else: # use poly values and its derivatives
        for i, xp in enumerate(indeces):
            res = np.copy(poly_vals[:num_pnts, xp[0]])
            # res = deepcopy(poly_vals[:num_pnts, xp[0]])
            for n in range(1, l):
                res *= poly_vals[num_pnts*n : num_pnts*(n+1), xp[n]]

            if IsTypeGood:
                A[:num_pnts, i] = res
            else:
                A.append(res)

            if ToGenDiff:
                for dl in range(l): # all derivatives
                    if 0 == dl: # it's the diff var
                        res = np.copy(poly_diff_vals[:num_pnts, xp[0]])
                    else:
                        res = np.copy(poly_vals[:num_pnts, xp[0]])

                    for n in range(1, l):
                        if n == dl: # it's the diff var
                            res *= poly_diff_vals[num_pnts*n : num_pnts*(n+1), xp[n]]
                        else:
                            res *= poly_vals[  num_pnts*n : num_pnts*(n+1), xp[n]]

                    if IsTypeGood:
                        A[num_pnts*(dl+1) : num_pnts*(dl+2), i] = res
                    else:
                        A.append(res)


    if not IsTypeGood:
        A = np.vstack(A).T

    norm = False
    if norm:
        As = A
        A  = []
        for i, e in enumerate(As):
            A.append(e/np.linalg.norm(e, 2))
        A = np.vstack(A)

    return A


@jit
def GenFunc(coeffs, l, poly=None, poly_diff=None, debug=False, pow_p=1, ToGenDiff=False):
    """
    Generates a function that gives RH using given points and solution
    coeffs -- solution of the system
    l -- dimension
    """
    import sympy as sp
    n_size = len(coeffs)
    xs = ['x' + str(i) for i in xrange(l)]
    vars_f = sp.symbols(' '.join(xs))
    A_symb = GenMat(n_size, np.array([vars_f]), poly=poly,
                poly_diff=poly_diff, debug=debug, pow_p=pow_p, ToGenDiff=ToGenDiff )

    params = ','.join(xs)
    if ToGenDiff:
        res = [ eval( "lambda {}: {}".format(params, i.dot(coeffs)) ) for i in A_symb ]
    else:
        f_l = "lambda {}: {}".format(params, A_symb[0, :].dot(coeffs))
        res = eval(f_l)
        # print f_l
    return res


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

def RenormXAndIdx(res, x, full=False):
    cf = x.shape[0]
    exists_idx = np.array(list(set( res % cf  )))
    resnew = np.zeros( (len(res), 2 ), dtype=int)
    for idx, i in enumerate(res):
        n, pos = divmod(i, cf)
        resnew[idx, 0] = np.where(exists_idx==pos)[0][0]
        resnew[idx, 1] = n

    if full:
        exists_idx = np.hstack((exists_idx, \
                                np.setdiff1d(xrange(x.shape[0]), exists_idx)  ))
        #                        np.array(list(set(range(x.shape[0])) - set(exists_idx) )) ))
    return resnew, x[exists_idx, :]


def PlotPoints(res, xout, fn='points', display=True):
    # plt.clf()
    # plt.hold()
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
        # plt.hold()
        plt.scatter(xout[pos, 0], xout[pos, 1], facecolors=color, s=ss_circ, alpha=0.3, edgecolors='face', hold=True)
        # plt.hold()
    plt.savefig(fn + '.pdf', bbox_inches='tight')
    if display:
        plt.show()



def indices_L(indices, subscript):
    """
    Filter indices for Sobol'
    Typically, only 1st indices are needed, so variable subscript contains only one element
    """
    indices = np.array(indices, copy=False)
    subscript = np.array(subscript, copy=False)
    all_idx = np.arange(indices.shape[1], dtype=int)
    other_idx = np.setdiff1d(all_idx, subscript)

    def filter_func(x):
        return (x[subscript] != 0).all() and (x[other_idx] == 0).all()

    return np.array([i for i, x in enumerate(indices) if filter_func(x)])




def SobolCoeffs(sol, l=2, func_norm=None, indices=None, subscripts=None):
    """
    Calculate Sobol' constants
    sol -- coefficients of poly expansion
    l -- dimension
    """


    if func_norm  is None:
        func_norm  = lambda n : 1.0
    if indices    is None:
        indices    = list(indeces_K_cut(l, len(sol)))
    if subscripts is None:
        subscripts = np.arange(l, dtype=int)[:, None]

    idx_L =  [ indices_L(indices, subscripts[i]) for i in range(l) ]


    # not used (now)
    def Li(i):
        resL = []
        for nj, j in enumerate(indices):
            if j[i] > 0 and max(  list(j[:i]) + list(j[i+1:])  ) == 0:
                resL.append(nj)
        return resL

    # L_*
    denom = 0.0
    for ni, i in enumerate(sol):
        if ni > 0:
            denom += i*i*np.prod(  [func_norm(j) for j in indices[ni] ] )

    # L_i
    # res = np.zeros(l, dtype=float)
    res = np.zeros(len(idx_L), dtype=float)
    for i, idx in enumerate(idx_L):
        # nom = 0.0
        # for j in Li(i):
        # for j in idx
            # nom += (sol[j]**2)*np.prod(  [func_norm(k) for k in indices[j] ] )

        # res[i] = nom
        res[i] = np.sum( [(sol[j]**2)*np.prod(  [ func_norm(k) for k in indices[j] ] ) for j in idx] )

    return res/denom



if __name__ == '__main__':
    print ('Test run')
    print (legendre(1, 2), legendre.normed(1, 2), legendre.snorm(1))

    exit(0)
    # num_p = 4 # number of points we select from on each axis.
    l = 2
    num_pnts = 10
    A_size = 5 # number of columns in matrix (numb. of monoms)
    # x_many = np.random.rand((num_p+1)**l, l)
    x_many = np.random.rand(num_pnts, l)
    A  = GenMat(A_size, x_many, poly=cheb)
    A2 = GenMat(A_size, x_many, poly=[cheb]*l)

    print (A)
    print (A2)
    print (A-A2)

    exit(0)


    from scipy.special.orthogonal import h_roots
    import rect_maxvol

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
    res, _ = rect_maxvol.rect_maxvol(A, minK=A_size, maxK=A_size, tol=1.0, start_maxvol_iters=10000)
    A = A[res, :]

    # remove unnecessary x
    res, x = RenormXAndIdx(res, x_many)
    PlotPoints(res, x)


