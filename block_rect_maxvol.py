from __future__ import print_function

import numpy.linalg as la
# from ids import SingularError
# from pyDOE import *
from numba import njit

from ids import *

# jit = lambda x : x
to_print_progress = False


def DebugPrint(s):
    if to_print_progress:
        print(s)
        sys.stdout.flush()


#   --- Service functions ---
# @jit
def form_permute(C, j, ind):  # REMOVE THIS
    C[ind], C[j] = C[j], C[ind]


# @jit
def mov_row(C, j, ind_x):  # REMOVE THIS
    C[[ind_x, j], :] = C[[j, ind_x], :]


def matrix_prep(A, ndim):
    n = A.shape[0]
    return A[np.arange(n).reshape(ndim, n // ndim).ravel(order='F')]


def row_to_block_indices(row_indices, block_size):
    block_indx = row_indices[::block_size] // block_size
    return block_indx


# C = A\hat{A}^{-1} fast recalculation with the help of SWM formula
@jit
def swm_recalculation(B, ndim, i, j):
    tmp_columns = np.copy(B[:, j:j + ndim])
    tmp_columns[j:j + ndim] -= np.eye(ndim)
    tmp_columns[i:i + ndim] += np.eye(ndim)

    b = B[i:i + ndim][:, j:j + ndim]

    tmp_rows = np.copy(B[i:i + ndim])
    tmp_rows[:, j:j + ndim] -= np.eye(ndim)

    B -= np.dot(tmp_columns, np.dot(np.linalg.inv(b), tmp_rows))
    return B


def change_intersept(inew, iold):
    """
    change two sets of rows or columns when indices may intercept with preserving order
    RETURN two sets of indices, then say A[idx_n] = A[idx_o]
    """

    # DebugPrint(str(inew) + '<->' + str(iold))
    union = np.array(list(set(inew) | set(iold)))
    idx_n = np.hstack((inew, np.setdiff1d(union, inew, assume_unique=True)))
    idx_o = np.hstack((iold, np.setdiff1d(union, iold, assume_unique=True)))
    return idx_n.astype(int), idx_o.astype(int)


@njit
def cold_start_tens(S, proj_matrix, ndim):
    num_block = proj_matrix.shape[0] // ndim
    for i in range(num_block):
        S[i, :, :] = proj_matrix[i * ndim:i * ndim + ndim] @ proj_matrix[i * ndim:i * ndim + ndim].T
    return 0


# --- Core algorithm functions ---
@jit
def block_maxvol(A_init, nder, tol=0.05, max_iters=100, swm_upd=True, debug=False):
    n, m = A_init.shape
    dim = nder + 1
    curr_det = None
    if swm_upd:
        A = A_init
        ids = A_init[:m]
        B = np.dot(A_init, np.linalg.inv(ids))
    else:
        A = np.copy(A_init)
        ids = A[:m]
        B = np.dot(A, np.linalg.inv(ids))

    possible_to_swap_blocks = True
    P = np.arange(n)
    index = np.zeros(2, dtype=int)
    iters = 0

    while possible_to_swap_blocks and (iters < max_iters):
        max_det = 1.0
        for k in range(m, n, dim):
            pair = B[k:k + dim]
            for j in range(0, m, dim):
                curr_det = np.abs(np.linalg.det(pair[:, j:j + dim]))
                if curr_det > max_det:
                    max_det = curr_det
                    index[0] = k
                    index[1] = j

        if max_det > (1 + tol):
            # Forming new permutation array
            for idx in range(dim):
                form_permute(P, index[1] + idx, index[0] + idx)

            if debug:
                print(P[:m])
                if swm_upd:
                    print('on the {} iteration with swm, pair {} {} chosen and pair{}'.format(iters, index[0], index[1],
                                                                                              B[
                                                                                              index[0]:index[0] + dim][
                                                                                              :,
                                                                                              index[1]:index[1] + dim]))
                if not swm_upd:
                    print(
                        'on the {} iteration with stan.oper, pair {} {} chosen and pair{}'.format(iters, index[0],
                                                                                                  index[1],
                                                                                                  B[
                                                                                                  index[0]:index[
                                                                                                               0] + dim][
                                                                                                  :, index[1]:index[
                                                                                                                  1] + dim]))
            # Recalculating with new rows position
            if swm_upd:
                B = swm_recalculation(B, dim, index[0], index[1])
                # for idx in range(dim):
                # mov_row(A,index[1] + idx,index[0] + idx)

            else:
                for idx in range(dim):
                    mov_row(A, index[1] + idx, index[0] + idx)
                B = np.dot(A, np.linalg.inv(ids))

            iters += 1
        else:
            possible_to_swap_blocks = False
    return P, B


@jit
def rect_block_core(init_proj_matrix, perm_vec, dim, Kmax, t=0.05):
    n, m = init_proj_matrix.shape
    k = Kmax // dim
    possible_to_add_block = True
    block_index = m // dim
    S = np.empty((n // dim, dim, dim))
    proj_matrix_expanded = np.empty((n, Kmax))
    proj_matrix_expanded[:, :m] = init_proj_matrix
    proj_matrix = proj_matrix_expanded[:, :m]
    cold_start_tens(S, proj_matrix, dim)
    while possible_to_add_block and block_index < k:
        det_list = [la.det(np.eye(dim) + S[i, :, :]) for i in range(n // dim)]
        elem = np.argmax(det_list[block_index:]) + block_index

        if det_list[elem] > (1 + t):
            range_j_dim = np.arange(block_index * dim, (block_index + 1) * dim)
            range_new_block = np.arange(elem * dim, elem * dim + dim)

            S[[block_index, elem], :, :] = S[[elem, block_index], :, :]
            indx_n, indx_o = change_intersept(range_j_dim, range_new_block)

            perm_vec[indx_n] = perm_vec[indx_o]
            proj_matrix[indx_n] = proj_matrix[indx_o]

            # ------ update part -----
            block = np.eye(dim) + proj_matrix[range_j_dim].dot(proj_matrix[range_j_dim].T)
            op3 = proj_matrix.dot(la.solve(block, proj_matrix[range_j_dim]).T)
            op4 = np.dot(op3, proj_matrix[range_j_dim])

            proj_matrix -= op4
            proj_matrix_expanded[:, block_index * dim:(block_index + 1) * dim] = op3
            block_index += 1

            proj_matrix = proj_matrix_expanded[:, :block_index * dim]
            cold_start_tens(S, proj_matrix, dim)
        else:
            # print('No relevant elements found')
            possible_to_add_block = False
    return proj_matrix, S, perm_vec


@jit
def rect_block_maxvol(A, nder, Kmax, rect_tol=0.05, tol=0.0, debug=False):
    assert (A.shape[1] % (nder + 1) == 0)
    assert (A.shape[0] % (nder + 1) == 0)
    assert (Kmax % (nder + 1) == 0)
    assert ((Kmax <= A.shape[0]) and (Kmax >= A.shape[1]))
    # DebugPrint("Start")

    try:
        pluq_perm, lu = plu_ids(A, nder, overwrite_a=False)
    except:
        pluq_perm, q, lu, inf = pluq_ids(A, nder, do_pullback=False, pullbacks=40, debug=False, overwrite_a=False)
        DebugPrint("ids.pluq_ids_index finishes")

    A = A[pluq_perm]  # [:, q]
    DebugPrint("block_maxvol starting")
    perm, C = block_maxvol(A, nder, tol=tol, max_iters=200, swm_upd=True)
    DebugPrint("block_maxvol finishes")

    bm_perm = pluq_perm[perm]
    DebugPrint("rect_block_maxvol_core starts")
    a, b, final_perm = rect_block_core(C, bm_perm, nder + 1, Kmax, t=rect_tol)
    DebugPrint("rect_block_maxvol_core finishes")
    return final_perm
