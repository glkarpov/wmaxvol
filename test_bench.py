import numpy as np
import numpy.linalg as la
from gen_mat import *
# from sympy import *
import sympy as sp
from export_f_txt import FindDiff, symb_to_func, MakeDiffs, SymbVars
from pyDOE import *
from numba import jit
import gen_mat as gen
from scipy.spatial.distance import pdist


@jit
def NumOfClusters(pnts, tol=0.005, full=True):
    """
    if full return array of clusters, else returns just its number
    """
    pd = pdist(pnts)

    n = pnts.shape[0]
    clutsts = []
    all_close = set()
    ci = 0
    for i in range(n-1):
        for j in range(i+1, n):
            if pd[ci] < tol:

                all_close |= {i, j}
                for c in clutsts:
                    if i in c or j in c:
                        c |= {i, j}
                        break
                else:
                    clutsts.append({i, j})

            ci += 1

    # all_close = reduce(lambda a, b: a|b, clutsts)
    for i in range(n):
        if i not in all_close:
            clutsts.append({i})

    return clutsts if full else len(clutsts)


def PrintMat(A):
    for i in A:
        print (' '.join([str(j) for j in i]))


def LSM (points, l, func, bounds=(-1.0, 1.0), poly = gen.cheb, pow_p=1, with_diff=False):
    """
    returns inf error of LSM approximation of the func rhs in the given points with number of monoms l
    """
    A = gen.GenMat(l, points, poly = poly, pow_p=pow_p, ToGenDiff=with_diff)
    if with_diff:
        rhs_val = RHS(func)
    else:
        rhs_val = func(points)

    coef = LA.solve(A.T.dot(A), A.T.dot(rhs_val))

    approx = approximant(points.shape[1], coef, poly, pow_p=pow_p, use_jit=True)
    # test = bounds[0] + (bounds[1] - bounds[0])*np.random.rand(int(1e5),points.shape[1])
    test_pnts = test_points_gen(int(1e5), points.shape[1], interval=bounds, distrib='random')
    error = error_est(rhs, approx, test_pnts)
    return error


def MakeValsAndNorms(funcs, pnts):
    res = []
    for f in funcs:
        vals = f(*pnts.T)
        norm = np.linalg.norm(vals, np.inf)
        res.append((f, vals, norm))

    return res

# @jit
def LebesgueConst(pnts, l, poly=cheb, test_pnts=None, pow_p=1, funcs=None):
    A = GenMat(l, pnts, poly=poly, debug=False, pow_p=pow_p, ToGenDiff=False)
    dim = pnts.shape[1]
    if test_pnts is None:
        test_pnts = test_points_gen(int(1e5), dim)

    ABig = GenMat(l, test_pnts, poly=poly, debug=False, pow_p=pow_p, ToGenDiff=False)
    # F = A.dot(np.linalg.solve(A.T.dot(A), ABig.T)) # Slower
    F = np.linalg.pinv(A).T.dot(ABig.T)

    maxx = np.max(np.sum(np.abs(F), axis=0))
    if funcs is not None:
        res = np.empty(1 + len(funcs))
        res[0] = maxx
        rs = 0
        for f, fvals, fnorm in funcs:
            rs += 1
            res[rs] = np.linalg.norm(F.T.dot(f(*pnts.T)) - fvals, np.inf)/fnorm

        return res
    else:
        return  maxx


# Numpy does not like it
def LebesgueConstSymb(pnts, l, poly=cheb, test_pnts=None, pow_p=1):
    A = GenMat(l, pnts, poly=poly, debug=False, pow_p=pow_p, ToGenDiff=False)
    dim = pnts.shape[1]
    if test_pnts is None:
        test_pnts = test_points_gen(int(1e5), dim)

    components = SymbVars(dim)
    al = approximant_list(dim, l, poly=poly, pow_p=pow_p, use_jit=False)
    # A_cross = np.linalg.solve(A.T.dot(A), A.T)
    F = np.linalg.pinv(A).T.dot(al(*components)) # now it's a sympy obj

    for i, e in enumerate(F):
        F[i] = sp.simplify(e)

    # func_poly = jit(sp.utilities.lambdify(components, F, 'numpy'))
    func_poly = sp.utilities.lambdify(components, F, 'numpy')

    if False:
        vrs = StrVars(pnts.shape[1])
        ff = 'lambda func_poly : lambda {0}: np.array(func_poly({0}))'.format(vrs)
        func_poly = np.vectorize(eval(ff)(func_poly), signature='(),()->(n)')
        vals = func_poly(*test_pnts.T)
        maxx = np.max(np.sum(np.abs(vals), axis=1))
    else:
        maxx = 0.0
        for pnt in test_pnts:
            cur_sum = np.sum(np.abs(func_poly(*pnt)))
            maxx = max(cur_sum, maxx)

    return maxx


def approximant_list(dim, l, poly=cheb, pow_p=1, use_jit=True):
    components = SymbVars(dim)
    sym_monoms = GenMat(l, np.array([components]), poly=poly, debug=False, pow_p=pow_p, ToGenDiff=False)[0]
    res = sp.utilities.lambdify(components, sym_monoms, 'numpy')
    if use_jit:
        res = jit(res)
    return res


def approximant(dim, coef, poly=cheb, pow_p=1, use_jit=True):
    # components = symbols(' '.join(['x' + str(comp_iter) for comp_iter in xrange(dim)]))\n",
    components = SymbVars(dim)
    sym_monoms = GenMat(coef.shape[0], np.array([components]), poly=poly, debug=False, pow_p=pow_p, ToGenDiff=False)[0]
    evaluate = np.dot(sym_monoms, coef)
    evaluate = sp.simplify(evaluate)
    res = sp.utilities.lambdify(components, evaluate, 'numpy')
    if use_jit:
        res = jit(res)
    return res

def error_est(origin_func, approx, points, norm=np.inf):
    error = la.norm(origin_func(*points.T) - approx(*points.T), norm) / la.norm(origin_func(*points.T), norm)
    return error


def test_points_gen(n_test, nder, interval=(-1.0,1.0), distrib='random'):
    return {'random' : lambda n_test, nder : (interval[1] - interval[0])*np.random.rand(n_test, nder) + interval[0],
            'LHS'    : lambda n_test, nder : (interval[1] - interval[0])*lhs(nder, samples=n_test) + interval[0]  }[distrib](n_test, nder)


def RHS(function, points):
    """
    Form RH-side from function and its derivative
    """

    nder = points.shape[1]
    nder1 = nder + 1
    block_rhs = np.empty(nder1*points.shape[0], dtype=points.dtype)
    block_rhs[::nder1] = function(*points.T)
    for j in range(nder):
        block_rhs[j+1::nder1] = function.diff[j](*points.T)

    return block_rhs


@jit
def NumOfClusters(pnts, tol=0.005, full=True):
    """
    if full return array of clusters, else returns just its number
    """
    pd = pdist(pnts)

    n = pnts.shape[0]
    clutsts = []
    all_close = set()
    ci = 0
    for i in range(n-1):
        for j in range(i+1, n):
            if pd[ci] < tol:

                all_close |= {i, j}
                for c in clutsts:
                    if i in c or j in c:
                        c |= {i, j}
                        break
                else:
                    clutsts.append({i, j})

            ci += 1

    # all_close = reduce(lambda a, b: a|b, clutsts)
    for i in range(n):
        if i not in all_close:
            clutsts.append({i})

    return clutsts if full else len(clutsts)

# ------------------------------
# -------------------- Various funcs ---------------

def gauss_sp(x,y):
    return 2*sp.exp(-((x**2)/2. + (y**2)/2.))

def sincos_sp(x,y):
    return (sp.sin((x**2)/2. - (y**2)/4. + 3) * sp.cos(2*x + 1 - sp.exp(y)))

def rosenbrock_sp(x,y):
    return ((1 - x)**2 + 100*(y - x**2)**2)

def roots_sp(x,y):
    return (sp.sqrt((x+2)**2 + (y+3)**2))

def f_quadro_3(x,y,z):
    return (2*((x**2)/2. + (y**2)/2. + (z**2)/2.))

def many_dim_sp(x,y,z,a,b):
    func = sp.sin(x+y+z) + a*b
    return func

def linear_sp(x,y):
    return (5*x + 2*y)



f_gauss      = symb_to_func(gauss_sp,      2, True, False, name='Gauss')
f_sincos     = symb_to_func(sincos_sp,     2, True, False, name='Sincos')
f_rosenbrock = symb_to_func(rosenbrock_sp, 2, True, False, name='Rosenbrock')
f_roots      = symb_to_func(roots_sp,      2, True, False, name='Roots')
f_many_dim   = symb_to_func(many_dim_sp,   5, True, False, name='Myltivariate')
f_linear     = symb_to_func(linear_sp,     2, True, False, name='Linear')


f_gauss.diff      = MakeDiffs(gauss_sp, 2)
f_sincos.diff     = MakeDiffs(sincos_sp, 2)
f_rosenbrock.diff = MakeDiffs(rosenbrock_sp, 2)
f_roots.diff      = MakeDiffs(roots_sp, 2)
f_linear.diff     = MakeDiffs(linear_sp, 2, True)
f_many_dim.diff   = MakeDiffs(many_dim_sp, 5, True)
f_quadro_3.diff   = MakeDiffs(f_quadro_3, 3, True)


