from __future__ import print_function
import numpy as np
import numpy.linalg as la
from ids import *
#from ids import SingularError
# from pyDOE import *
from numba import jit
import sys
import math

# jit = lambda x : x
to_print_progress = False


def DebugPrint(s):
    if to_print_progress:
        print(s)
        sys.stdout.flush()


###-------------- Service functions ----------------- ###
# @jit
def form_permute(C, j, ind):  # REMOVE THIS
    C[ind], C[j] = C[j], C[ind]


# @jit
def mov_row(C, j, ind_x):  # REMOVE THIS
    C[[ind_x, j], :] = C[[j, ind_x], :]


def matrix_prep(A, ndim):
    n = A.shape[0]
    return A[np.arange(n).reshape(ndim, n // ndim).ravel(order='F')]


# C = A\hat{A}^{-1} fast recalculation with the help of SWM formula
@jit
def SWM(B, ndim, i, j):
    tmp_columns = np.copy(B[:, j:j + ndim])
    tmp_columns[j:j + ndim] -= np.eye(ndim)
    tmp_columns[i:i + ndim] += np.eye(ndim)

    b = B[i:i + ndim][:, j:j + ndim]

    tmp_rows = np.copy(B[i:i + ndim])
    tmp_rows[:, j:j + ndim] -= np.eye(ndim)

    B -= np.dot(tmp_columns, np.dot(np.linalg.inv(b), tmp_rows))
    return (B)


def change_intersept(inew, iold):
    """
    change two sets of rows or columns when indices may intersept with preserving order
    RETURN two sets of indices,
    than say A[idx_n] = A[idx_o]
    """

    # DebugPrint(str(inew) + '<->' + str(iold))
    union = np.array(list(set(inew) | set(iold)))
    idx_n = np.hstack((inew, np.setdiff1d(union, inew, assume_unique=True)))
    idx_o = np.hstack((iold, np.setdiff1d(union, iold, assume_unique=True)))
    return idx_n.astype(int), idx_o.astype(int)


# stuff to handle with matrix linings. Puts matrix U in the lining of A, i.e. : B = A*UA (default) or B = AUA*.
class lining:
    def __init__(self, A, U, inv=False):
        if not inv:
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
# @jit
def rect_core(C, C_sigma, ndim):
    inv_block = la.inv(np.eye(ndim) + np.dot(C_sigma, C_sigma.conjugate().T))
    puzzle = lining(C_sigma, inv_block)
    U = np.hstack((np.eye(C.shape[1]) - puzzle.assemble(), puzzle.left_prod()))
    C_new = lining(C, U, inv=True).left_prod()
    return C_new, puzzle.assemble()


# @jit
def cold_start(C, ndim):
    n = C.shape[0]
    values = []
    for i in range(0, n, ndim):
        CC_T = np.dot(C[i:i + ndim], C[i:i + ndim].conjugate().T)
        values.append(CC_T)
    return values


@jit
def cold_start_tens(C, ndim):
    num_block = C.shape[0] // ndim
    S = np.empty((num_block, ndim, ndim))
    for i in range(num_block):
        S[i, :, :] = np.dot(C[i * ndim:i * ndim + ndim], C[i * ndim:i * ndim + ndim].T)
    return (S)


### ------------------------------------
def erase_init(func, x, nder, r):
    func.x = x
    func.nder = nder
    func.r = r


@jit
def point_erase(p_chosen, p, C):
    ndim = point_erase.nder + 1
    P = np.copy(p)
    ### P - perm vector. [::ndim]//ndim encodes point. 
    for point_idx in p_chosen[::(ndim)] // (ndim):
        erase_inx = []
        for j in P[::(ndim)] // (ndim):
            if j not in p_chosen[::(ndim)] // (ndim):
                if np.linalg.norm(point_erase.x[point_idx] - point_erase.x[j], 2) < point_erase.r:
                    elem = np.where(P == j * ndim)[0]
                    for idx in range(ndim):
                        erase_inx.append(elem + idx)
        P = np.delete(P, erase_inx)
        C = np.delete(C, erase_inx, axis=0)
    return (P, C)


# @jit
# def weights_update(
### -------------- Core algorithm functions -------------- ###
@jit
def block_maxvol(A_init, nder, tol=0.05, max_iters=100, swm_upd=True, debug=False):
    n, m = A_init.shape
    ndim = nder + 1
    if swm_upd:
        A = A_init
        ids = A_init[:m]
        B = np.dot(A_init, np.linalg.inv(ids))
    else:
        A = np.copy(A_init)
        ids = A[:m]
        B = np.dot(A, np.linalg.inv(ids))

    curr_det = np.abs(np.linalg.det(ids))
    Fl = True
    P = np.arange(n)
    index = np.zeros((2), dtype=int)
    iters = 0

    while Fl and (iters < max_iters):
        max_det = 1.0
        for k in range(m, n, ndim):
            pair = B[k:k + ndim]
            for j in range(0, m, ndim):
                curr_det = np.abs(np.linalg.det(pair[:, j:j + ndim]))
                if curr_det > max_det:
                    max_det = curr_det
                    index[0] = k
                    index[1] = j

        if (max_det) > (1 + tol):
            # Forming new permutation array
            for idx in range(ndim):
                form_permute(P, index[1] + idx, index[0] + idx)

            if debug == True:
                print(P[:m])
            if (swm_upd == True) and (debug == True):
                print('on the {} iteration with swm, pair {} {} chosen and pair{}'.format(iters, index[0], index[1],
                                                                                          B[index[0]:index[0] + ndim][:,
                                                                                          index[1]:index[1] + ndim]))
            if (swm_upd == False) and (debug == True):
                print(
                    'on the {} iteration with stan.oper, pair {} {} chosen and pair{}'.format(iters, index[0], index[1],
                                                                                              B[
                                                                                              index[0]:index[0] + ndim][
                                                                                              :, index[1]:index[
                                                                                                              1] + ndim]))
            ### Recalculating with new rows position
            if swm_upd == True:
                B = SWM(B, ndim, index[0], index[1])
                # for idx in range(ndim):
                # mov_row(A,index[1] + idx,index[0] + idx)

            else:
                for idx in range(ndim):
                    mov_row(A, index[1] + idx, index[0] + idx)
                B = np.dot(A, np.linalg.inv(ids))

            iters += 1
        else:
            Fl = False
    return (P, B)


# @jit
def rect_block_maxvol_core(A_init, P, nder, Kmax, t=0.05, to_erase=None):
    ndim = nder + 1
    M, n = A_init.shape
    block_n = M // ndim  # Whole amount of blocks in matrix A
    # P = p #np.arange(M) # Permutation vector
    Fl = True
    Fl_cs = True
    ids_init = A_init[:n]
    temp_init = np.dot(A_init, np.linalg.pinv(ids_init))
    C = np.copy(temp_init)

    shape_index = n
    CC_sigma = []

    while Fl and (shape_index < Kmax):

        block_index = shape_index // ndim

        if Fl_cs:
            CC_sigma = cold_start(C, ndim)
            Fl_cs = False

        ind_array = la.det(np.eye(ndim) + CC_sigma)
        elem = np.argmax(ind_array[block_index:]) + block_index

        if (ind_array[elem] > 1 + t):
            CC_sigma[block_index], CC_sigma[elem] = CC_sigma[elem], CC_sigma[block_index]
            for idx in range(ndim):
                form_permute(P, shape_index + idx, elem * ndim + idx)
                mov_row(C, shape_index + idx, elem * ndim + idx)
            C_new, line = rect_core(C, C[shape_index:shape_index + ndim], ndim)
            if to_erase is not None:
                ###----------------------------------
                P, C_new = to_erase(P[shape_index:shape_index + ndim], P, C_new)
                ###----------------------------------
            ### update list of CC_sigma
            # for k in range(len(CC_sigma)):
            # CC_sigma[k] = CC_sigma[k] - np.dot(C_w[k*ndim:ndim*(k+1)], np.dot(line, C_w[k*ndim:ndim*(k+1)].conjugate().T))
            C = C_new
            CC_sigma = cold_start(C, ndim)
        else:
            print('No relevant elements found')
            Fl = False

        shape_index += ndim
    return (C, CC_sigma, P)


@jit
def rect_block_core(C, P, nder, Kmax, t=0.05, to_erase=None):
    ndim = nder + 1
    n, m = C.shape
    num_block = n // ndim
    k = Kmax // ndim
    Fl = True
    block_index = m // ndim

    S = cold_start_tens(C, ndim)

    # C_new = np.empty((n, Kmax))
    # C_new[:, :m] = C
    while Fl and block_index < k:
        # C = C_new[:, :block_index*ndim]

        # det_list = [la.det(np.eye(ndim) + S[i,:,:]) for i in range(num_block)]
        det_list = la.det(np.eye(ndim) + S)
        elem = np.argmax(det_list[block_index:]) + block_index

        if det_list[elem] > (1 + t):
            range_j_dim = np.arange(block_index * ndim, block_index * ndim + ndim)
            range_new_block = np.arange(elem * ndim, elem * ndim + ndim)

            S[[block_index, elem], :, :] = S[[elem, block_index], :, :]
            indx_n, indx_o = change_intersept(range_j_dim, range_new_block)

            P[indx_n] = P[indx_o]
            C[indx_n] = C[indx_o]

            # ------ update part -----
            # inv_block = la.inv(np.eye(ndim) + C[range_j_dim].dot(C[range_j_dim].T))
            # op3 = C.dot(C[range_j_dim].T.dot(inv_block))

            block = np.eye(ndim) + C[range_j_dim].dot(C[range_j_dim].T)
            op3 = C.dot(la.solve(block, C[range_j_dim]).T)
            op4 = np.dot(op3, C[range_j_dim])

            # for i in range(block_index, num_block):
            #   i_ndim = i*ndim
            #    S[i,:,:] -= np.dot(op4[i_ndim:i_ndim + ndim],C[i_ndim:i_ndim + ndim].T)

            C = np.hstack((C - op4, op3))
            if to_erase is not None:
                ###----------------------------------
                P, C = to_erase(P[range_j_dim], P, C)
                ###----------------------------------
            S = cold_start_tens(C, ndim)
            # C -= op4
            # C_new[:, block_index*ndim:block_index*(ndim)+ndim] = op3
            block_index += 1
        else:
            # print('No relevant elements found')
            Fl = False
    return (C, S, P)


# @jit
def rect_block_maxvol(A, nder, Kmax, max_iters, rect_tol=0.05, tol=0.0, debug=False, to_erase=None):
    assert (A.shape[1] % (nder + 1) == 0)
    assert (A.shape[0] % (nder + 1) == 0)
    assert (Kmax % (nder + 1) == 0)
    assert ((Kmax <= A.shape[0]) and (Kmax >= A.shape[1]))
    DebugPrint("Start")

    try:
        pluq_perm, lu = plu_ids(A, nder, overwrite_a=False)
    except:
        pluq_perm, q, lu, inf = pluq_ids(A, nder, do_pullback=False, pullbacks=40, debug=False, overwrite_a=False)
        DebugPrint("ids.pluq_ids_index finishes")

    A = A[pluq_perm]  # [:, q]
    DebugPrint("block_maxvol starting")
    perm, C = block_maxvol(A, nder, tol=tol, max_iters=200, swm_upd=True)
    DebugPrint("block_maxvol finishes")

    A = A[perm]
    bm_perm = pluq_perm[perm]
    ### Perform erasure after we got initial points in square matrix
    if to_erase is not None:
        # A1 = np.copy(A)
        bm_perm, C = to_erase(bm_perm[:C.shape[1]], bm_perm, C)
    # else:
    #    A1 = A    
    DebugPrint("rect_block_maxvol_core starts")
    # a, b, final_perm = rect_block_maxvol_core(A, bm_perm, nder, Kmax, t = rect_tol, to_erase = to_erase)
    a, b, final_perm = rect_block_core(C, bm_perm, nder, Kmax, t=rect_tol, to_erase=to_erase)
    DebugPrint("rect_block_maxvol_core finishes")
    return (final_perm)
