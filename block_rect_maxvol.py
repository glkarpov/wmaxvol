import numpy as np
import numpy.linalg as la
import ids
from block_maxvol import *
from gen_mat import *
from sympy import *
from export_f_txt import FindDiff, symb_to_func
from pyDOE import *

# stuff to handle with matrix linings. Puts matrix U in lining, i.e. : B = A*UA or B = AUA*.
class lining:
    def __init__(self, A, U,inv = False):
        if inv == False:
            self.left = A.conjugate().T
            self.right = A
            self.core = U
        else:
            self.left = A
            self.right = A.conjugate().T
            self.core = U
    def left_prod(self):
        return np.dot(self.left, self.core)
    def right_prod(self):
        return np.dot(self.core, self.right)
    def assemble(self):
        return np.dot(self.left, self.right_prod())

# main func to form new coeff matrix
def rect_core(C, C_sigma, ndim):
    inv_block = la.inv(np.eye(ndim) + np.dot(C_sigma, C_sigma.conjugate().T))
    puzzle = lining(C_sigma,inv_block)
    U = np.hstack((np.eye(C.shape[1]) - puzzle.assemble(), puzzle.left_prod()))
    C_new = lining(C,U,inv = True).left_prod()
    return C_new, puzzle.assemble()

def form_permute(C, j, ind): #  REMOVE THIS
    C[ind],C[j]=C[j],C[ind]


def mov_row(C, j, ind_x): #  REMOVE THIS
    C[[ind_x,j],:] = C[[j,ind_x],:]


def cold_start(C, ndim):
    n = C.shape[0]
    k = n // ndim
    values = []
    for i in range(0,k):
        CC_T = np.dot(C[i*ndim:i*ndim+ndim], C[i*ndim:i*ndim+ndim].conjugate().T)
        values.append((CC_T))
    return values


def matrix_prep(A, ndim, l):
    return A[ np.arange(l*ndim).reshape(ndim, l).flatten(order='F') , :]


def rect_block_maxvol_core(A_init, nder, Kmax, t = 0.05):
    ndim = nder + 1
    M, n = A_init.shape
    block_n = M // ndim # Whole amount of blocks in matrix A
    P = np.arange(M) # Permutation vector
    Fl = True
    Fl_cs = True
    ids_init = A_init[:n]
    temp_init = np.dot(A_init, np.linalg.pinv(ids_init))

    A = np.copy(A_init)
    A_hat = np.copy(ids_init)
    C = np.copy(temp_init)

    shape_index = n
    C_w = np.copy(C)
    CC_sigma = []

    while (Fl == True) and (shape_index < Kmax):

        if (Fl_cs == False):
            ind_array = la.det(np.eye(ndim) + CC_sigma)
            elem = np.argmax(np.abs(ind_array[(shape_index // ndim):])) + (shape_index // ndim)
            print elem
            if (ind_array[elem] > 1 + t):
                CC_sigma[shape_index/ndim], CC_sigma[elem] = CC_sigma[elem], CC_sigma[shape_index/ndim]
                for idx in range(ndim):
                    form_permute(P,shape_index + idx,elem*ndim + idx)
                    mov_row(C_w,shape_index + idx,elem*ndim + idx)
                C_new, line = rect_core(C_w,C_w[shape_index:shape_index + ndim],ndim)
                #print C_new.shape, C_w.shape

                ### update list of CC_sigma
                for k in range(block_n):
                    CC_sigma[k] = CC_sigma[k] - lining(C_w[k*ndim:ndim*(k+1)],line,inv=True).assemble()
                C_w = C_new
                shape_index += ndim
            else:
                print ('elements not found')
                Fl = False
                
        if Fl_cs:
            CC_sigma = cold_start(C_w, ndim)
            ind_array = la.det(np.eye(ndim) + CC_sigma)
            elem = np.argmax(np.abs(ind_array[(shape_index // ndim):])) + (shape_index // ndim)
            print elem
            if (ind_array[elem] > 1 + t):
                CC_sigma[shape_index/ndim], CC_sigma[elem] = CC_sigma[elem], CC_sigma[shape_index/ndim]
                for idx in range(ndim):
                    form_permute(P,shape_index + idx,elem*ndim + idx)
                    mov_row(C_w,shape_index + idx,elem*ndim + idx)
            else:
                print ('cold_start fail')
                Fl = False
            shape_index += ndim
            C_new, line = rect_core(C_w,C_w[n:shape_index],ndim)
            
            ### update list of CC_sigma
            for k in range(block_n):
                CC_sigma[k] = CC_sigma[k] - np.dot(C_w[k*ndim:ndim*(k+1)], np.dot(line, C_w[k*ndim:ndim*(k+1)].conjugate().T))
            C_w = C_new 
            Fl_cs = False
            

    return (C_w, CC_sigma, P)
    
def rect_block_maxvol(A, nder, Kmax, max_iters, rect_tol = 0.05, tol = 0.0, debug = False, ext_debug = False):
    assert (A.shape[1] % (nder+1) == 0)
    assert (A.shape[0] % (nder+1) == 0)
    assert (Kmax % (nder+1) == 0)
    assert ((Kmax <= A.shape[0]) and (Kmax >= A.shape[1]))
    pluq_perm,l,u,q,inf = ids.pluq_ids(A,nder, debug=False)
    A_init = np.dot(ids.perm_matrix(pluq_perm),np.dot(A,ids.perm_matrix(q)))
    A_rect_init,_,perm = block_maxvol(A_init, nder, tol = tol,max_iters=200,swm_upd=True)
    bm_perm = ids.perm_array(np.dot(ids.perm_matrix(perm),ids.perm_matrix(pluq_perm)))
    a, b, c = rect_block_maxvol_core(A_rect_init,nder,Kmax,t = rect_tol)
    final_perm = ids.perm_array(np.dot(ids.perm_matrix(c),ids.perm_matrix(bm_perm)))
    
    if ext_debug:
        return (a,b, final_perm, bm_perm, pluq_perm)
    else:
        return (final_perm)

def test(A,x,x_test, nder, col_expansion, N_rows, function):
    N_column = col_expansion*(nder+1)
    M = A[:, :N_column]
    #print la.matrix_rank(M)
    #print N_column
    
    piv = rect_block_maxvol(M, nder, Kmax = N_rows, max_iters=100, rect_tol = 0.05, tol = 0.0, debug = False, ext_debug = False)
    cut_piv = piv[:N_rows]
    
    block_func_deriv = RHS(function,nder,x)
    c_block, res_x, rank, s = np.linalg.lstsq(M[cut_piv],block_func_deriv[cut_piv])
    
    taken_p = x[cut_piv[::(nder+1)]/(nder+1),:]
    approx_calcul = approximant(nder,c_block)
    error = error_est(function, approx_calcul, x_test)
    
    l_bound = np.amin(x)
    u_bound = np.amax(x)
    plt.xlim(l_bound-0.15, u_bound+0.15)
    plt.ylim(l_bound-0.15, u_bound+0.15)
    plt.plot(taken_p[:,0],taken_p[:,1], 'b^', label = "BMV")
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=2, borderaxespad=0.)
    plt.figtext(.8, .8, "E = {}".format(error))
    plt.grid(True)
    fn = 'func={}_d={}_num={}_nder={}.pdf'.format(function.__name__, N_column, N_rows, nder)
    plt.savefig(fn)
    return error

def approximant(nder, coef):
    components = symbols(' '.join(['x' + str(comp_iter) for comp_iter in xrange(nder)]))
    sym_monoms = GenMat(coef.shape[0], np.array([components]),poly=cheb, debug=False,pow_p=1,ToGenDiff=False)
    evaluate = np.dot(sym_monoms[0], coef)
    evaluate = simplify(evaluate)
    res = utilities.lambdify(components, evaluate, 'numpy')
    return res

def test_points_gen(n_test, nder, distrib = 'random'):
    if distrib == 'random' :
        x_test = 2*np.random.rand(n_test, nder) - 1
    if distrib == 'LHS' :
        x_test = lhs(nder, samples=n_test)
    return x_test
     
def error_est(origin_func, approx, points):
    error = la.norm(origin_func(*points.T) - approx(*points.T), np.inf) / la.norm(origin_func(*points.T), np.inf)
    return error

### returns 2 values - function on domain, and block structured
def gauss_sp(x,y):
    return 2*exp(-((x**2)/2. + (y**2)/2.))


def quadro_3(x,y,z):
    return (2*((x**2)/2. + (y**2)/2. + (z**2)/2.))

def many_dim_sp(x,y,z,a,b):
    func = sin(x+y+z) + a*b
    return func

def linear_sp(x,y):
    return (5*x + x**2 + y**2)

# print 'Initializations'

# quadro_3  = symb_to_func(quadro_3,    3, True, False)
gauss     = symb_to_func(gauss_sp,    2, True, False)
many_dim  = symb_to_func(many_dim_sp, 5, True, False)
linear    = symb_to_func(linear_sp,   2, True, False)

# print 'Funcs Got'

gauss.diff    = [FindDiff(gauss_sp,    2, i, False) for i in range(1,3)]
many_dim.diff = [FindDiff(many_dim_sp, 5, i, False) for i in range(1,6)]
linear.diff   = [FindDiff(linear,      2, i, False) for i in range(1,3)]
quadro_3.diff = [FindDiff(quadro_3,    3, i, False) for i in range(1,4)]

def RHS(function, points):
    """
    Form RH-side from function and its derivative
    """

    """
    old realization
    func = function(*points.T)
    func_diff = [function.diff[i](*points.T) for i in range(len(function.diff))]
    block_rhs = np.zeros((nder+1)*(points.shape[0]))
    for i in range(points.shape[0]):
        block_rhs[i*(nder+1)] = func[i]
        for j in range(nder):
            block_rhs[i*(nder+1)+j+1] = func_diff[j][i]

    """

    nder= points.shape[1]
    nder1 = nder + 1
    block_rhs = np.empty(nder1*points.shape[0], dtype=points.dtype)
    block_rhs[::nder1] = function(*points.T)
    for j in range(nder):
        block_rhs[j+1::nder1] = function.diff[j](*points.T)

    return block_rhs


