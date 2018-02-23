import numpy as np
import numpy.linalg as la
import ids
from ids import SingularError
from block_maxvol import *
from gen_mat import *
from export_f_txt import FindDiff, symb_to_func, MakeDiffs, SymbVars
from pyDOE import *
from numba import jit
import sys
from test_bench import *

# jit = lambda x : x
to_print_progress = False

def DebugPrint(s):
    if to_print_progress:
        print s
        sys.stdout.flush()

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
@jit  
def rect_core(C, C_sigma, ndim):
    inv_block = la.inv(np.eye(ndim) + np.dot(C_sigma, C_sigma.conjugate().T))
    puzzle = lining(C_sigma,inv_block)
    U = np.hstack((np.eye(C.shape[1]) - puzzle.assemble(), puzzle.left_prod()))
    C_new = lining(C,U,inv = True).left_prod()
    return C_new, puzzle.assemble()

@jit
def form_permute(C, j, ind): #  REMOVE THIS
    C[ind],C[j]=C[j],C[ind]

@jit
def mov_row(C, j, ind_x): #  REMOVE THIS
    C[[ind_x,j],:] = C[[j,ind_x],:]

@jit
def cold_start(C, ndim):
    n = C.shape[0]
    values = []
    for i in range(0, n, ndim):
        CC_T = np.dot(C[i:i + ndim], C[i:i + ndim].conjugate().T)
        values.append((CC_T))
    return values


def matrix_prep(A, ndim):
    n = A.shape[0]
    return A[ np.arange(n).reshape(ndim, n // ndim).ravel(order='F') ]

@jit
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

    while Fl and (shape_index < Kmax):

        shape_index_unit = shape_index // ndim
        if Fl_cs:
            CC_sigma = cold_start(C_w, ndim)
            ind_array = la.det(np.eye(ndim) + CC_sigma)
            elem = np.argmax(np.abs(ind_array[shape_index_unit:])) + shape_index_unit
            #print elem
            if (ind_array[elem] > 1 + t):
                CC_sigma[shape_index_unit], CC_sigma[elem] = CC_sigma[elem], CC_sigma[shape_index_unit]
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


        else:
            ind_array = la.det(np.eye(ndim) + CC_sigma)
            elem = np.argmax(np.abs(ind_array[shape_index_unit:])) + shape_index_unit
            #print elem
            if (ind_array[elem] > 1 + t):
                CC_sigma[shape_index_unit], CC_sigma[elem] = CC_sigma[elem], CC_sigma[shape_index_unit]
                for idx in range(ndim):
                    form_permute(P, shape_index + idx, elem*ndim + idx)
                    mov_row(C_w, shape_index + idx, elem*ndim + idx)
                C_new, line = rect_core(C_w, C_w[shape_index:shape_index + ndim], ndim)
                #print C_new.shape, C_w.shape

                ### update list of CC_sigma
                for k in range(block_n):
                    CC_sigma[k] = CC_sigma[k] - lining(C_w[k*ndim:ndim*(k+1)], line, inv=True).assemble()
                C_w = C_new
                shape_index += ndim
            else:
                print ('elements not found')
                Fl = False



    return (C_w, CC_sigma, P)



# @jit
def rect_block_maxvol(A, nder, Kmax, max_iters, rect_tol = 0.05, tol = 0.0, debug = False, ext_debug = False):
    assert (A.shape[1] % (nder+1) == 0)
    assert (A.shape[0] % (nder+1) == 0)
    assert (Kmax % (nder+1) == 0)
    assert ((Kmax <= A.shape[0]) and (Kmax >= A.shape[1]))
    DebugPrint ("Start")

    pluq_perm, l, u, q, inf = ids.pluq_ids_index(A, nder, debug=False)
    DebugPrint ("ids.pluq_ids_index finishes")

    # A_init = np.dot(ids.perm_matrix(pluq_perm), np.dot(A, ids.perm_matrix(q)))
    A = A[pluq_perm][:, q]
    DebugPrint ("block_maxvol starting")
    A, _, perm = block_maxvol(A, nder, tol = tol, max_iters=200, swm_upd=True)
    DebugPrint ("block_maxvol finishes")

    # bm_perm = ids.perm_array(np.dot(ids.perm_matrix(perm), ids.perm_matrix(pluq_perm)))
    bm_perm = pluq_perm[perm]
    DebugPrint ("rect_block_maxvol_core starts")
    a, b, c = rect_block_maxvol_core(A, nder, Kmax, t = rect_tol)
    DebugPrint ("rect_block_maxvol_core finishes")

    # final_perm = ids.perm_array(np.dot(ids.perm_matrix(c), ids.perm_matrix(bm_perm)))
    final_perm = bm_perm[c]

    if ext_debug:
        return (a,b, final_perm, bm_perm, pluq_perm)
    else:
        return (final_perm)

def test(A, x, x_test, nder, col_expansion, N_rows, functions, poly=cheb, to_save_pivs=True, to_export_pdf=True, fnpdf=None):
    N_column = col_expansion*(nder+1)
    M = A[:, :N_column]
    #print la.matrix_rank(M)
    #print N_column
    
    if to_save_pivs:
        pivs = rect_block_maxvol(M, nder, Kmax = N_rows, max_iters=100, rect_tol = 0.05, tol = 0.0, debug = False, ext_debug = False)
        test.pivs = pivs
        test.N_column = N_column
    else:
        try:
            pivs = test.pivs
            assert test.N_column == N_column, "Call test with to_save_pivs=True first"
        except:
            assert False, "Call test with to_save_pivs=True first"

    assert pivs.size >= N_rows, "Wrong N_rows value"
    cut_piv = pivs[:N_rows]
    taken_indices = cut_piv[::(nder+1)] // (nder+1)

    if type(functions) is not list:
        functions = [functions]

    error = np.empty(len(functions), dtype=float)

    for i, function in enumerate(functions):
        block_func_deriv = RHS(function, x[taken_indices])
        c_block, res_x, rank, s = np.linalg.lstsq(M[cut_piv], block_func_deriv)
    
        approx_calcul = approximant(nder, c_block, poly=poly)
        error[i] = error_est(function, approx_calcul, x_test)
    
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

    return error, taken_indices


