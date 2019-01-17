from __future__ import print_function
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
from sobol_lib import *
from block_rect_maxvol import *
from gen_points import *
import sys
import matplotlib.pyplot as plt
import matplotlib
from mpl_toolkits.mplot3d import Axes3D
import re
from matplotlib import cm

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

# obsolete
def LSM (points, l, func, bounds=(-1.0, 1.0), poly = gen.cheb, pow_p=1, with_diff=False):
    """
    returns inf error of LSM approximation of the func rhs in the given points with number of monoms l
    """
    A = gen.GenMat(l, points, poly = poly, pow_p=pow_p, ToGenDiff=with_diff)
    if with_diff:
        rhs_val = RHS(func, points)
    else:
        rhs_val = func(points)

    coef = LA.solve(A.T.dot(A), A.T.dot(rhs_val))

    approx = approximant(points.shape[1], coef, poly, pow_p=pow_p, use_jit=True)
    # test = bounds[0] + (bounds[1] - bounds[0])*np.random.rand(int(1e5),points.shape[1])
    test_pnts = test_points_gen(int(1e5), points.shape[1], interval=bounds, distrib='random')
    error = error_est(rhs_val, approx, test_pnts)
    return error


def MakeValsAndNorms(funcs, pnts):
    res = []
    for f in funcs:
        vals = f(*pnts.T)
        norm = np.linalg.norm(vals, np.inf)
        res.append((f, vals, norm))

    return res

# @jit
def LebesgueConst(pnts, l, poly=cheb, test_pnts=None, pow_p=1, funcs=None, derivative = True):
    A = GenMat(l, pnts, poly=poly, debug=False, pow_p=pow_p, ToGenDiff=derivative)
    
    nder = pnts.shape[1]
    if derivative:
        A = matrix_prep(A, nder+1)
    if test_pnts is None:
        test_pnts = test_points_gen(int(1e5), nder)

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
            rhs = RHS(f, pnts, derivative = derivative)
            res[rs] = np.linalg.norm(F.T.dot(rhs) - fvals, np.inf)/fnorm

        return res
    else:
        return  maxx


# Numba does not like it
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


# Not used
def approximant_list(dim, l, poly=cheb, pow_p=1, use_jit=True):
    components = SymbVars(dim)
    sym_monoms = GenMat(l, np.array([components]), poly=poly, debug=False, pow_p=pow_p, ToGenDiff=False)[0]
    res = sp.utilities.lambdify(components, sym_monoms, 'numexpr')
    if use_jit:
        res = jit(res)
    return res


def approximant(dim, coef, poly=cheb, pow_p=1, use_jit=True):
    components = SymbVars(dim)
    sym_monoms = GenMat(coef.shape[0], np.array([components]), poly=poly, debug=False, pow_p=pow_p, ToGenDiff=False)[0]
    evaluate = np.dot(sym_monoms, coef)
    evaluate = sp.simplify(evaluate)

    if use_jit:
        res = jit(sp.utilities.lambdify(components, evaluate))
    else:
        # res = sp.utilities.lambdify(components, evaluate, 'numpy')
        res = sp.utilities.lambdify(components, evaluate, 'numexpr')

    return res

def error_est(origin_func, approx, points, norm=np.inf):
    error = la.norm(origin_func(*points.T) - approx(*points.T), norm) / la.norm(origin_func(*points.T), norm)
    return error



def RHS(function, points, derivative = True):
    """
    Form RH-side from function and its derivative
    """
    nder = points.shape[1]
    if derivative == True:
        ndim = nder + 1
        rhs = np.empty(ndim*points.shape[0], dtype=points.dtype)
        rhs[::ndim] = function(*points.T)
        for j in range(nder):
            rhs[j+1::ndim] = function.diff[j](*points.T)
    else:
        rhs = np.empty(points.shape[0], dtype=points.dtype)
        rhs[:] = function(*points.T)
    return rhs


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

def gauss_double_sp(x,y):
    return 2*sp.exp(-(((x+0.9)**2)/2. + ((y-0.4)**2)/2.)) + 2*sp.exp(-(((x-0.9)**2)/2. + ((y+0.4)**2)/2.))

def sincos_sp(x,y):
    return (sp.sin((x**2)/2. - (y**2)/4. + 3) * sp.cos(2*x + 1 - sp.exp(y)))

def rosenbrock_sp(x,y):
    return ((1 - x)**2 + 100*(y - x**2)**2)

def roots_sp(x,y):
    return (sp.sqrt((x+2)**2 + (y+3)**2))

def ellipse_sp(x,y):
    b,e = 0.2,0.95
    sigma,n = 0.1, 10
    r = sp.sqrt(x**2 + y**2)
    phi = sp.atan2(y,x)
    R = r*sp.sqrt(1 - (e*sp.cos(phi))**2)/b
    return((sp.exp(-1*(R**2)/(2*(sigma**2))))*sp.cos(n*phi))

def f_quadro_3(x,y,z):
    return (2*((x**2)/2. + (y**2)/2. + (z**2)/2.))

def many_dim_sp(x,y,z,a,b):
    func = sp.sin(x+y+z) + a*b
    return func

def linear_sp(x,y):
    return (5*x + 2*y)

def branin_sp(x,y):
    a = 1
    b = 5.1/(4*((np.pi)**2))
    c = 5/np.pi
    r, s, t = 6, 10, 1/(8*np.pi)
    return (a*(y - b*(x**2) + c*x - r)**2 + s*(1 - t)*sp.cos(x) + s)

def holsclaw_sp(x,y):
    return (sp.log(1.05 + x + x**2 + x*y))

def piston(M,S,V_0,k,P_0,T_a,T_0):
    M   = ((M + 1)*30)/2. + 30
    S   = ((S + 1)*0.015)/2. + 0.005
    V_0 = ((V_0 + 1)*0.008)/2. + 0.002
    k   = ((k + 1)*4000)/2. + 1000
    P_0 = ((P_0 + 1)*20000)/2. + 90000
    T_a = ((T_a + 1)*6)/2. + 290
    T_0 = ((T_0 + 1)*20)/2. + 340

    A = P_0*S + 19.62*M - k*V_0/S
    V = S/(2*k)*(np.sqrt(A**2 + 4*k*P_0*V_0*T_a/T_0) - A)
    C = 2*np.pi*np.sqrt(M/(k + S**2*P_0*V_0*T_a/(T_0*V**2)))
    return C

f_gauss      = symb_to_func(gauss_sp,      2, True, False, name='Gauss')
f_gauss_doubl= symb_to_func(gauss_double_sp,2,True, False, name='Gauss_doubl')
f_sincos     = symb_to_func(sincos_sp,     2, True, False, name='Sincos')
f_rosenbrock = symb_to_func(rosenbrock_sp, 2, True, False, name='Rosenbrock')
f_roots      = symb_to_func(roots_sp,      2, True, False, name='Roots')
f_many_dim   = symb_to_func(many_dim_sp,   5, True, False, name='Myltivariate')
f_linear     = symb_to_func(linear_sp,     2, True, False, name='Linear')
f_branin     = symb_to_func(branin_sp,     2, True, False, name='Branin')
f_holsclaw   = symb_to_func(holsclaw_sp,   2, True, False, name='Holsclaw')
f_ellipse    = symb_to_func(ellipse_sp,    2, True, False, name='Ellipse')

f_gauss.diff      = MakeDiffs(gauss_sp, 2)
f_gauss_doubl.diff= MakeDiffs(gauss_double_sp, 2)
f_sincos.diff     = MakeDiffs(sincos_sp, 2)
f_rosenbrock.diff = MakeDiffs(rosenbrock_sp, 2)
f_roots.diff      = MakeDiffs(roots_sp, 2)
f_linear.diff     = MakeDiffs(linear_sp, 2, True)
f_branin.diff     = MakeDiffs(branin_sp, 2)
f_holsclaw.diff   = MakeDiffs(holsclaw_sp, 2)
f_ellipse.diff    = MakeDiffs(ellipse_sp,  2)
f_many_dim.diff   = MakeDiffs(many_dim_sp, 5, True)
f_quadro_3.diff   = MakeDiffs(f_quadro_3, 3, True)


def test_bm(A, x, nder, col_expansion, N_rows, cut_radius = 0.15, to_save_pivs=True, to_export_pdf=True, fnpdf=None):
    N_column = col_expansion*(nder+1)
    M = A[:, :N_column]
    if to_save_pivs:
        if cut_radius == None:
            to_erase = None
        else:    
            erase_init(point_erase, x, nder, r = cut_radius)
            to_erase = point_erase
        pivs = rect_block_maxvol(M, nder, Kmax = N_rows, max_iters=100, rect_tol = 0.05, tol = 0.0, debug = False, to_erase = to_erase)
        test_bm.pivs = pivs
        test_bm.N_column = N_column
    else:
        try:
            pivs = test_bm.pivs
            assert test_bm.N_column == N_column, "Call test with to_save_pivs=True first"
        except:
            assert False, "Call test with to_save_pivs=True first"

    assert pivs.size >= N_rows, "Wrong N_rows value"
    cut_piv = pivs[:N_rows]
    taken_indices = cut_piv[::(nder+1)] // (nder+1)
    
    #if nder == 2 and (fnpdf is not None or to_export_pdf):
        #l_bound = np.amin(x, 0)
        #u_bound = np.amax(x, 0)
        #delta = (u_bound - l_bound)/20.0
        #fig = plt.figure()
        #plt.xlim(l_bound[0] - delta[0], u_bound[0] + delta[0])
        #plt.ylim(l_bound[1] - delta[1], u_bound[1] + delta[1])
        #plt.plot(x[taken_indices, 0], x[taken_indices, 1], 'b^')
        ##plt.title("E = {}".format(error))
        #plt.grid(True)
        #if fnpdf is None:
            #fnpdf = 'func={}_columns={}_rows={}_pnts={}.pdf'.format(function.__name__, N_column, N_rows, len(taken_indices))
            #fnpdf = 'columns={}_rows={}.pdf'.format(N_column, N_rows)
        ##print("Num of points = {}, saving to file {}".format(len(taken_indices), fnpdf))
        #plt.savefig(fnpdf)
        #plt.close(fig)

    return taken_indices



#------------------- Visualisation ----------

def file_extraction(Filepath, new_extr = True):
    if new_extr:
        srch = re.compile(r'([\d\s]+)_Nrows=(\d+)_expans=(\d+)')
        fnd = srch.findall(open(Filepath, 'r').read())
        return tuple(np.array(i) for i in zip(*[(int(i1), int(i2), [int(p) for p in im1.strip().split(' ') if len(p) > 0])
                                                for im1, i1, i2 in fnd]))
    else:
        srch = re.compile(r'([\d\s]+)_error=([\+\-\d\.eE]+)_Nrows=(\d+)_expans=(\d+)')
        fnd = srch.findall(open(Filepath, 'r').read())
        return tuple(np.array(i) for i in zip(*[(float(i0), int(i1), int(i2), [int(p) for p in im1.strip().split(' ') if len(p) > 0])
                                                for im1, i0, i1, i2 in fnd]))


def DataToMesh(error, N_row, N_col, *args):
    row_s = sorted(list(set(N_row)))
    col_s = sorted(list(set(N_col)))
    data = {(N_row[i], N_col[i]) : e for i, e in enumerate(error)}
    
    res = np.empty((len(row_s), len(col_s)), dtype=float)
    for i, r in enumerate(row_s):
        for j, c in enumerate(col_s):
            try:
                res[i,j] = data[(r, c)]
            except:
                res[i,j] = np.nan
    X, Y = np.meshgrid(row_s, col_s)
    return res.T, X, Y


def PlotError(fn, log_it=False):
    error, N_row, N_col = DataToMesh(*file_extraction(fn))
    if log_it:
        error = np.log10(error)

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_surface(N_row, N_col, error, edgecolor='black', linewidth=0.5, cmap = cm.Spectral)
    # ax.legend()
    ax.set_xlabel('N_rows', fontsize=10)
    ax.set_ylabel('N_columns')
    plt.show()


def PlotPoints(fn, row, col, x, to_save_fig=False, fn_out="func.pdf"):
    _, N_row, N_col, idx = file_extraction(fn)

    l_bound = np.amin(x, 0)
    u_bound = np.amax(x, 0)
    delta = (u_bound - l_bound)/20.0
    plt.xlim(l_bound[0] - delta[0], u_bound[0] + delta[0])
    plt.ylim(l_bound[1] - delta[1], u_bound[1] + delta[1])
    plt.plot(x[idx, 0], x[idx, 1], 'b^')
    # plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=2, borderaxespad=0.)
    # plt.title("E = {}".format(error))
    plt.grid(True)
    if to_save_fig:
        # fn = 'func={}_d={}_num={}_nder={}.pdf'.format(function.__name__, N_column, N_rows, nder)
        plt.savefig(fn_out)


def PlotError_3D(N_row, N_col, error_ext, log_it=False):
    error, N_row, N_col = DataToMesh(error_ext, N_row, N_col)
    
    if log_it:
        error = np.log10(error)
    
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_surface(N_row, N_col, error, edgecolor='black', linewidth=0.5, cmap = cm.Spectral)
    # ax.legend()
    ax.set_xlabel('N_rows', fontsize=10)
    ax.set_ylabel('N_columns')
    plt.show()
    

if __name__ == '__main__':
    if len(sys.argv) > 1:
        fn = sys.argv[1]
    else:
        fn = "func=Rosenbrock_poly=cheb.txt"

    PlotError(fn, log_it=False)