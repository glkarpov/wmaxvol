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


def test_points_gen(n_test, nder, interval=(-1.0, 1.0), distrib='random', **kwargs):
    return {'random' : lambda n_test, nder : (interval[1] - interval[0])*np.random.rand(n_test, nder) + interval[0],\
            'lhs'    : lambda n_test, nder : (interval[1] - interval[0])*lhs(nder, samples=n_test, **kwargs) + interval[0],\
            'halton' : lambda n_test, nder : (interval[1] - interval[0])*halton(nder, n_test, **kwargs) + interval[0],\
            'sobol'  : lambda n_test, nder : (interval[1] - interval[0])*GenSobol(nder, n_test, **kwargs) + interval[0]\
            }[distrib.lower()](n_test, nder)


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



def halton(dim, n, start=0, bases=None):
    if bases is None:
        bases = primes

    r = np.zeros([n, dim])

    prime_inv_s = 1.0 / bases[:dim]

    for k, i in enumerate(range(start, start + n)):

        t = np.full(dim, i, dtype=int)

        prime_inv = np.copy(prime_inv_s)
        while 0 < np.sum(t):
            for j in range (0, dim):
                d = t[j] % bases[j]
                r[k, j] += d * prime_inv[j]
                prime_inv[j] /= bases[j]
                t[j] //= bases[j]

    return r

def GenSobol(dim = 2, N = 200,  seed = 0):
    res = np.empty((N, dim), dtype=float)
    for i in range(N):
        res[i, :], seed = i4_sobol ( dim, seed )

    return res


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

def branin_sp(x,y):
    a = 1
    b = 5.1/(4*((np.pi)**2))
    c = 5/np.pi
    r, s, t = 6, 10, 1/(8*np.pi)
    return (a*(y - b*(x**2) + c*x - r)**2 + s*(1 - t)*sp.cos(x) + s)

def holsclaw_sp(x,y):
    return (sp.log(1.05 + x + x**2 + x*y))

f_gauss      = symb_to_func(gauss_sp,      2, True, False, name='Gauss')
f_sincos     = symb_to_func(sincos_sp,     2, True, False, name='Sincos')
f_rosenbrock = symb_to_func(rosenbrock_sp, 2, True, False, name='Rosenbrock')
f_roots      = symb_to_func(roots_sp,      2, True, False, name='Roots')
f_many_dim   = symb_to_func(many_dim_sp,   5, True, False, name='Myltivariate')
f_linear     = symb_to_func(linear_sp,     2, True, False, name='Linear')
f_branin     = symb_to_func(branin_sp,     2, True, False, name='Branin')
f_holsclaw   = symb_to_func(holsclaw_sp,   2, True, False, name='Holsclaw')

f_gauss.diff      = MakeDiffs(gauss_sp, 2)
f_sincos.diff     = MakeDiffs(sincos_sp, 2)
f_rosenbrock.diff = MakeDiffs(rosenbrock_sp, 2)
f_roots.diff      = MakeDiffs(roots_sp, 2)
f_linear.diff     = MakeDiffs(linear_sp, 2, True)
f_branin.diff     = MakeDiffs(branin_sp, 2)
f_holsclaw.diff   = MakeDiffs(holsclaw_sp, 2)
f_many_dim.diff   = MakeDiffs(many_dim_sp, 5, True)
f_quadro_3.diff   = MakeDiffs(f_quadro_3, 3, True)


primes = np.array([
        2,    3,    5,    7,   11,   13,   17,   19,   23,   29, \
       31,   37,   41,   43,   47,   53,   59,   61,   67,   71, \
       73,   79,   83,   89,   97,  101,  103,  107,  109,  113, \
      127,  131,  137,  139,  149,  151,  157,  163,  167,  173, \
      179,  181,  191,  193,  197,  199,  211,  223,  227,  229, \
      233,  239,  241,  251,  257,  263,  269,  271,  277,  281, \
      283,  293,  307,  311,  313,  317,  331,  337,  347,  349, \
      353,  359,  367,  373,  379,  383,  389,  397,  401,  409, \
      419,  421,  431,  433,  439,  443,  449,  457,  461,  463, \
      467,  479,  487,  491,  499,  503,  509,  521,  523,  541, \
      547,  557,  563,  569,  571,  577,  587,  593,  599,  601, \
      607,  613,  617,  619,  631,  641,  643,  647,  653,  659, \
      661,  673,  677,  683,  691,  701,  709,  719,  727,  733, \
      739,  743,  751,  757,  761,  769,  773,  787,  797,  809, \
      811,  821,  823,  827,  829,  839,  853,  857,  859,  863, \
      877,  881,  883,  887,  907,  911,  919,  929,  937,  941, \
      947,  953,  967,  971,  977,  983,  991,  997, 1009, 1013, \
     1019, 1021, 1031, 1033, 1039, 1049, 1051, 1061, 1063, 1069, \
     1087, 1091, 1093, 1097, 1103, 1109, 1117, 1123, 1129, 1151, \
     1153, 1163, 1171, 1181, 1187, 1193, 1201, 1213, 1217, 1223, \
     1229, 1231, 1237, 1249, 1259, 1277, 1279, 1283, 1289, 1291, \
     1297, 1301, 1303, 1307, 1319, 1321, 1327, 1361, 1367, 1373, \
     1381, 1399, 1409, 1423, 1427, 1429, 1433, 1439, 1447, 1451, \
     1453, 1459, 1471, 1481, 1483, 1487, 1489, 1493, 1499, 1511, \
     1523, 1531, 1543, 1549, 1553, 1559, 1567, 1571, 1579, 1583, \
     1597, 1601, 1607, 1609, 1613, 1619, 1621, 1627, 1637, 1657, \
     1663, 1667, 1669, 1693, 1697, 1699, 1709, 1721, 1723, 1733, \
     1741, 1747, 1753, 1759, 1777, 1783, 1787, 1789, 1801, 1811, \
     1823, 1831, 1847, 1861, 1867, 1871, 1873, 1877, 1879, 1889, \
     1901, 1907, 1913, 1931, 1933, 1949, 1951, 1973, 1979, 1987, \
     1993, 1997, 1999, 2003, 2011, 2017, 2027, 2029, 2039, 2053, \
     2063, 2069, 2081, 2083, 2087, 2089, 2099, 2111, 2113, 2129, \
     2131, 2137, 2141, 2143, 2153, 2161, 2179, 2203, 2207, 2213, \
     2221, 2237, 2239, 2243, 2251, 2267, 2269, 2273, 2281, 2287, \
     2293, 2297, 2309, 2311, 2333, 2339, 2341, 2347, 2351, 2357, \
     2371, 2377, 2381, 2383, 2389, 2393, 2399, 2411, 2417, 2423, \
     2437, 2441, 2447, 2459, 2467, 2473, 2477, 2503, 2521, 2531, \
     2539, 2543, 2549, 2551, 2557, 2579, 2591, 2593, 2609, 2617, \
     2621, 2633, 2647, 2657, 2659, 2663, 2671, 2677, 2683, 2687, \
     2689, 2693, 2699, 2707, 2711, 2713, 2719, 2729, 2731, 2741, \
     2749, 2753, 2767, 2777, 2789, 2791, 2797, 2801, 2803, 2819, \
     2833, 2837, 2843, 2851, 2857, 2861, 2879, 2887, 2897, 2903, \
     2909, 2917, 2927, 2939, 2953, 2957, 2963, 2969, 2971, 2999, \
     3001, 3011, 3019, 3023, 3037, 3041, 3049, 3061, 3067, 3079, \
     3083, 3089, 3109, 3119, 3121, 3137, 3163, 3167, 3169, 3181, \
     3187, 3191, 3203, 3209, 3217, 3221, 3229, 3251, 3253, 3257, \
     3259, 3271, 3299, 3301, 3307, 3313, 3319, 3323, 3329, 3331, \
     3343, 3347, 3359, 3361, 3371, 3373, 3389, 3391, 3407, 3413, \
     3433, 3449, 3457, 3461, 3463, 3467, 3469, 3491, 3499, 3511, \
     3517, 3527, 3529, 3533, 3539, 3541, 3547, 3557, 3559, 3571, \
     3581, 3583, 3593, 3607, 3613, 3617, 3623, 3631, 3637, 3643, \
     3659, 3671, 3673, 3677, 3691, 3697, 3701, 3709, 3719, 3727, \
     3733, 3739, 3761, 3767, 3769, 3779, 3793, 3797, 3803, 3821, \
     3823, 3833, 3847, 3851, 3853, 3863, 3877, 3881, 3889, 3907, \
     3911, 3917, 3919, 3923, 3929, 3931, 3943, 3947, 3967, 3989, \
     4001, 4003, 4007, 4013, 4019, 4021, 4027, 4049, 4051, 4057, \
     4073, 4079, 4091, 4093, 4099, 4111, 4127, 4129, 4133, 4139, \
     4153, 4157, 4159, 4177, 4201, 4211, 4217, 4219, 4229, 4231, \
     4241, 4243, 4253, 4259, 4261, 4271, 4273, 4283, 4289, 4297, \
     4327, 4337, 4339, 4349, 4357, 4363, 4373, 4391, 4397, 4409, \
     4421, 4423, 4441, 4447, 4451, 4457, 4463, 4481, 4483, 4493, \
     4507, 4513, 4517, 4519, 4523, 4547, 4549, 4561, 4567, 4583, \
     4591, 4597, 4603, 4621, 4637, 4639, 4643, 4649, 4651, 4657, \
     4663, 4673, 4679, 4691, 4703, 4721, 4723, 4729, 4733, 4751, \
     4759, 4783, 4787, 4789, 4793, 4799, 4801, 4813, 4817, 4831, \
     4861, 4871, 4877, 4889, 4903, 4909, 4919, 4931, 4933, 4937, \
     4943, 4951, 4957, 4967, 4969, 4973, 4987, 4993, 4999, 5003, \
     5009, 5011, 5021, 5023, 5039, 5051, 5059, 5077, 5081, 5087, \
     5099, 5101, 5107, 5113, 5119, 5147, 5153, 5167, 5171, 5179, \
     5189, 5197, 5209, 5227, 5231, 5233, 5237, 5261, 5273, 5279, \
     5281, 5297, 5303, 5309, 5323, 5333, 5347, 5351, 5381, 5387, \
     5393, 5399, 5407, 5413, 5417, 5419, 5431, 5437, 5441, 5443, \
     5449, 5471, 5477, 5479, 5483, 5501, 5503, 5507, 5519, 5521, \
     5527, 5531, 5557, 5563, 5569, 5573, 5581, 5591, 5623, 5639, \
     5641, 5647, 5651, 5653, 5657, 5659, 5669, 5683, 5689, 5693, \
     5701, 5711, 5717, 5737, 5741, 5743, 5749, 5779, 5783, 5791, \
     5801, 5807, 5813, 5821, 5827, 5839, 5843, 5849, 5851, 5857, \
     5861, 5867, 5869, 5879, 5881, 5897, 5903, 5923, 5927, 5939, \
     5953, 5981, 5987, 6007, 6011, 6029, 6037, 6043, 6047, 6053, \
     6067, 6073, 6079, 6089, 6091, 6101, 6113, 6121, 6131, 6133, \
     6143, 6151, 6163, 6173, 6197, 6199, 6203, 6211, 6217, 6221, \
     6229, 6247, 6257, 6263, 6269, 6271, 6277, 6287, 6299, 6301, \
     6311, 6317, 6323, 6329, 6337, 6343, 6353, 6359, 6361, 6367, \
     6373, 6379, 6389, 6397, 6421, 6427, 6449, 6451, 6469, 6473, \
     6481, 6491, 6521, 6529, 6547, 6551, 6553, 6563, 6569, 6571, \
     6577, 6581, 6599, 6607, 6619, 6637, 6653, 6659, 6661, 6673, \
     6679, 6689, 6691, 6701, 6703, 6709, 6719, 6733, 6737, 6761, \
     6763, 6779, 6781, 6791, 6793, 6803, 6823, 6827, 6829, 6833, \
     6841, 6857, 6863, 6869, 6871, 6883, 6899, 6907, 6911, 6917, \
     6947, 6949, 6959, 6961, 6967, 6971, 6977, 6983, 6991, 6997, \
     7001, 7013, 7019, 7027, 7039, 7043, 7057, 7069, 7079, 7103, \
     7109, 7121, 7127, 7129, 7151, 7159, 7177, 7187, 7193, 7207, \
     7211, 7213, 7219, 7229, 7237, 7243, 7247, 7253, 7283, 7297, \
     7307, 7309, 7321, 7331, 7333, 7349, 7351, 7369, 7393, 7411, \
     7417, 7433, 7451, 7457, 7459, 7477, 7481, 7487, 7489, 7499, \
     7507, 7517, 7523, 7529, 7537, 7541, 7547, 7549, 7559, 7561, \
     7573, 7577, 7583, 7589, 7591, 7603, 7607, 7621, 7639, 7643, \
     7649, 7669, 7673, 7681, 7687, 7691, 7699, 7703, 7717, 7723, \
     7727, 7741, 7753, 7757, 7759, 7789, 7793, 7817, 7823, 7829, \
     7841, 7853, 7867, 7873, 7877, 7879, 7883, 7901, 7907, 7919, \
     7927, 7933, 7937, 7949, 7951, 7963, 7993, 8009, 8011, 8017, \
     8039, 8053, 8059, 8069, 8081, 8087, 8089, 8093, 8101, 8111, \
     8117, 8123, 8147, 8161, 8167, 8171, 8179, 8191, 8209, 8219, \
     8221, 8231, 8233, 8237, 8243, 8263, 8269, 8273, 8287, 8291, \
     8293, 8297, 8311, 8317, 8329, 8353, 8363, 8369, 8377, 8387, \
     8389, 8419, 8423, 8429, 8431, 8443, 8447, 8461, 8467, 8501, \
     8513, 8521, 8527, 8537, 8539, 8543, 8563, 8573, 8581, 8597, \
     8599, 8609, 8623, 8627, 8629, 8641, 8647, 8663, 8669, 8677, \
     8681, 8689, 8693, 8699, 8707, 8713, 8719, 8731, 8737, 8741, \
     8747, 8753, 8761, 8779, 8783, 8803, 8807, 8819, 8821, 8831, \
     8837, 8839, 8849, 8861, 8863, 8867, 8887, 8893, 8923, 8929, \
     8933, 8941, 8951, 8963, 8969, 8971, 8999, 9001, 9007, 9011, \
     9013, 9029, 9041, 9043, 9049, 9059, 9067, 9091, 9103, 9109, \
     9127, 9133, 9137, 9151, 9157, 9161, 9173, 9181, 9187, 9199, \
     9203, 9209, 9221, 9227, 9239, 9241, 9257, 9277, 9281, 9283, \
     9293, 9311, 9319, 9323, 9337, 9341, 9343, 9349, 9371, 9377, \
     9391, 9397, 9403, 9413, 9419, 9421, 9431, 9433, 9437, 9439, \
     9461, 9463, 9467, 9473, 9479, 9491, 9497, 9511, 9521, 9533, \
     9539, 9547, 9551, 9587, 9601, 9613, 9619, 9623, 9629, 9631, \
     9643, 9649, 9661, 9677, 9679, 9689, 9697, 9719, 9721, 9733, \
     9739, 9743, 9749, 9767, 9769, 9781, 9787, 9791, 9803, 9811, \
     9817, 9829, 9833, 9839, 9851, 9857, 9859, 9871, 9883, 9887, \
     9901, 9907, 9923, 9929, 9931, 9941, 9949, 9967, 9973,10007, \
    10009,10037,10039,10061,10067,10069,10079,10091,10093,10099, \
    10103,10111,10133,10139,10141,10151,10159,10163,10169,10177, \
    10181,10193,10211,10223,10243,10247,10253,10259,10267,10271, \
    10273,10289,10301,10303,10313,10321,10331,10333,10337,10343, \
    10357,10369,10391,10399,10427,10429,10433,10453,10457,10459, \
    10463,10477,10487,10499,10501,10513,10529,10531,10559,10567, \
    10589,10597,10601,10607,10613,10627,10631,10639,10651,10657, \
    10663,10667,10687,10691,10709,10711,10723,10729,10733,10739, \
    10753,10771,10781,10789,10799,10831,10837,10847,10853,10859, \
    10861,10867,10883,10889,10891,10903,10909,10937,10939,10949, \
    10957,10973,10979,10987,10993,11003,11027,11047,11057,11059, \
    11069,11071,11083,11087,11093,11113,11117,11119,11131,11149, \
    11159,11161,11171,11173,11177,11197,11213,11239,11243,11251, \
    11257,11261,11273,11279,11287,11299,11311,11317,11321,11329, \
    11351,11353,11369,11383,11393,11399,11411,11423,11437,11443, \
    11447,11467,11471,11483,11489,11491,11497,11503,11519,11527, \
    11549,11551,11579,11587,11593,11597,11617,11621,11633,11657, \
    11677,11681,11689,11699,11701,11717,11719,11731,11743,11777, \
    11779,11783,11789,11801,11807,11813,11821,11827,11831,11833, \
    11839,11863,11867,11887,11897,11903,11909,11923,11927,11933, \
    11939,11941,11953,11959,11969,11971,11981,11987,12007,12011, \
    12037,12041,12043,12049,12071,12073,12097,12101,12107,12109, \
    12113,12119,12143,12149,12157,12161,12163,12197,12203,12211, \
    12227,12239,12241,12251,12253,12263,12269,12277,12281,12289, \
    12301,12323,12329,12343,12347,12373,12377,12379,12391,12401, \
    12409,12413,12421,12433,12437,12451,12457,12473,12479,12487, \
    12491,12497,12503,12511,12517,12527,12539,12541,12547,12553, \
    12569,12577,12583,12589,12601,12611,12613,12619,12637,12641, \
    12647,12653,12659,12671,12689,12697,12703,12713,12721,12739, \
    12743,12757,12763,12781,12791,12799,12809,12821,12823,12829, \
    12841,12853,12889,12893,12899,12907,12911,12917,12919,12923, \
    12941,12953,12959,12967,12973,12979,12983,13001,13003,13007, \
    13009,13033,13037,13043,13049,13063,13093,13099,13103,13109, \
    13121,13127,13147,13151,13159,13163,13171,13177,13183,13187, \
    13217,13219,13229,13241,13249,13259,13267,13291,13297,13309, \
    13313,13327,13331,13337,13339,13367,13381,13397,13399,13411, \
    13417,13421,13441,13451,13457,13463,13469,13477,13487,13499 ])

def NextPrime(p):
    s = np.max(p) + 1
    while np.min(s % p) == 0:
        s += 1
    return s

def Primes(n):
    a = np.array([2])
    for _ in range(n-1):
        a = np.append(a, NextPrime(a))

    return a

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
    
    if nder == 2 and (fnpdf is not None or to_export_pdf):
        l_bound = np.amin(x, 0)
        u_bound = np.amax(x, 0)
        delta = (u_bound - l_bound)/20.0
        fig = plt.figure()
        plt.xlim(l_bound[0] - delta[0], u_bound[0] + delta[0])
        plt.ylim(l_bound[1] - delta[1], u_bound[1] + delta[1])
        plt.plot(x[taken_indices, 0], x[taken_indices, 1], 'b^')
        # plt.title("E = {}".format(error))
        plt.grid(True)
        if fnpdf is None:
            # fnpdf = 'func={}_columns={}_rows={}_pnts={}.pdf'.format(function.__name__, N_column, N_rows, len(taken_indices))
            fnpdf = 'columns={}_rows={}.pdf'.format(N_column, N_rows)
        # print "Num of points = {}, saving to file {}".format(len(taken_indices), fnpdf)
        plt.savefig(fnpdf)
        plt.close(fig)

    return taken_indices