import numpy as np
from numba import jit
import numpy.linalg as la
from maxvolpy.maxvol import rect_maxvol
from mva_test import *

def C_upd_alpha(C_old, ndim, block_indx, alpha):
    ix_range = np.arange(ndim*block_indx, ndim*(block_indx + 1))
    int_prod = np.eye(ndim) + (alpha**2 - 1) * np.dot(C_old[ix_range], C_old[ix_range].T)
    op1 = C_old @ ((alpha**2 - 1) * C_old[ix_range,:].T) @ la.inv(int_prod)
    op2 = op1 @ C_old[ix_range]
    C_new = (C_old - op2)
    C_new[:,ix_range] =  C_new[:,ix_range] * alpha
    return C_new

@jit
def bmaxvol_sequent_coeff(A, n_iter, bas_length=None, block_size=1, to_use_recalc = False):
    ndim = block_size
    n, m = A.shape
    if bas_length is None:
        bas_length = max(ndim, m)
    prod_shape = min(ndim, m)
    A_bas = np.copy(A[:bas_length])
    A_bas_base = np.copy(A_bas)
    c = np.dot(A, la.pinv(A_bas))

    block_indcs = np.arange(int(n // ndim))
    uniq_base_idx = np.copy(block_indcs[:(int(bas_length // ndim))])
    weights = np.ones(bas_length // ndim)
    S = cold_start_tens(c, ndim)

    for i in range(n_iter):
        det_list = la.det(np.eye(ndim) + S)
        elem = np.argmax(det_list)
        if elem in uniq_base_idx:
            ix = np.where(uniq_base_idx == elem)[0][0]
            weights[ix] += 1
            if to_use_recalc:
                C1 = np.copy(c)
                c = C_upd_alpha(C1, ndim, ix, np.sqrt(weights[ix]/(weights[ix]-1)))
            else:
                range_new_block = np.arange(ix * ndim, ix * ndim + ndim)
                A_bas[range_new_block, :] = np.sqrt(weights[ix]) * A_bas_base[range_new_block, :]
                c = np.dot(A, la.pinv(A_bas))
        else:
            range_new_block = np.arange(elem * ndim, elem * ndim + ndim)
            A_bas = np.vstack((A_bas, A[range_new_block]))
            A_bas_base = np.vstack((A_bas_base, A[range_new_block]))
            uniq_base_idx = np.append(uniq_base_idx, block_indcs[elem])
            weights = np.append(weights, np.array(1))
            c = np.dot(A, la.pinv(A_bas))
        S = cold_start_tens(c, ndim)

    return (uniq_base_idx, weights)


@jit
def cold_start_tens(C, ndim):
    m = C.shape[1]
    num_block = C.shape[0] // ndim
    if ndim < C.shape[1]:
        S = np.empty((num_block, ndim, ndim))
        for i in range(num_block):
            S[i, :, :] = np.dot(C[i * ndim:i * ndim + ndim], C[i * ndim:i * ndim + ndim].T)
    else:
        S = np.empty((num_block, m, m))
        for i in range(num_block):
            S[i, :, :] = np.dot(C[i * ndim:i * ndim + ndim].T, C[i * ndim:i * ndim + ndim])
    return S



def maxvol_precondition(A, pivs):
    k = A.shape[1]
    indc, _ = rect_maxvol(A, maxK=max_pts(k))
    idx_n, idx_o = change_intersept(np.arange(indc.shape[0]), indc)
    A[idx_n, :] = A[idx_o, :]
    pivs[idx_n] = pivs[idx_o]
    return A, pivs



def max_pts(m):
    return int(m * (m + 1) / 2)

def wmaxvol(A, n_iter, out_dim, do_mv_precondition = True):
    k = A.shape[1]
    n_points = int(A.shape[0] / out_dim)
    if do_mv_precondition:
        A, pivs = maxvol_precondition(A, np.arange(A.shape[0]))
    else:
        pivs = np.arange(n_points)
    pts, wts = bmaxvol_sequent_coeff(A, n_iter, bas_length=max_pts(k), block_size=out_dim)
    return pivs[pts], wts.astype(int)