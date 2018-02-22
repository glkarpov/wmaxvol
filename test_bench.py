import numpy as np
import numpy.linalg as la
from gen_mat import *
from sympy import *
from export_f_txt import FindDiff, symb_to_func, MakeDiffs, SymbVars
from pyDOE import *
from numba import jit
import sys

def LSM (points, bounds, l, poly = gen.cheb, rhs = rosenbrock):
    A = gen.GenMat(l, points, poly = poly, ToGenDiff=False)
    c = LA.solve(np.dot(A.T, A), np.dot(A.T, rhs(pnts)))

    approx = approximant(points.shape[1], c, poly)
    test = bounds[0] + (bounds[1] - bounds[0])*np.random.rand(int(1e5),points.shape[1])
    error = error_est(rhs, approx, points)
    return error

def approximant(nder, coef, poly=cheb):
    # components = symbols(' '.join(['x' + str(comp_iter) for comp_iter in xrange(nder)]))\n",
    components = SymbVars(nder)
    sym_monoms = GenMat(coef.shape[0], np.array([components]), poly=poly, debug=False, pow_p=1, ToGenDiff=False)
    evaluate = np.dot(sym_monoms[0], coef)
    evaluate = simplify(evaluate)
    res = utilities.lambdify(components, evaluate, 'numpy')
    return res

def error_est(origin_func, approx, points):
    error = la.norm(origin_func(*points.T) - approx(*points.T), np.inf) / la.norm(origin_func(*points.T), np.inf)
    return error

### returns 2 values - function on domain, and block structured
def gauss_sp(x,y):
    return 2*exp(-((x**2)/2. + (y**2)/2.))

def sincos_sp(x,y):
    return (sin((x**2)/2. - (y**2)/4. + 3) * cos(2*x + 1 - exp(y)))

def rosenbrock_sp(x,y):
    return ((1 - x)**2 + 100*(y - x**2)**2)

rosenbrock = symb_to_func(rosenbrock_sp, 2, True, False, name='Rosenbrock')

def roots_sp(x,y):
    return (sqrt((x+2)**2 + (y+3)**2))

def quadro_3(x,y,z):
    return (2*((x**2)/2. + (y**2)/2. + (z**2)/2.))

def many_dim_sp(x,y,z,a,b):
    func = sin(x+y+z) + a*b
    return func

def linear_sp(x,y):
    return (5*x + 2*y)