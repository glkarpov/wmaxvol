from maxvolpy.maxvol import rect_maxvol
from mva_test import *


def proj_matrix_upd(c_prev, ndim, block_indx, alpha):
    ix_range = np.arange(ndim * block_indx, ndim * (block_indx + 1))
    int_prod = np.eye(ndim) + (alpha ** 2 - 1) * np.dot(c_prev[ix_range], c_prev[ix_range].T)
    op1 = c_prev @ ((alpha ** 2 - 1) * c_prev[ix_range, :].T) @ la.inv(int_prod)
    op2 = op1 @ c_prev[ix_range]
    c_new = (c_prev - op2)
    c_new[:, ix_range] = c_new[:, ix_range] * alpha
    return c_new


@jit
def wmaxvol(a, n_iter, out_dim, bas_length=None, do_mv_precondition=False, to_use_recalc=False):
    n, m = a.shape
    n_points = int(n / out_dim)
    if do_mv_precondition:
        a, pivs = maxvol_precondition(a, np.arange(n))
    else:
        pivs = np.arange(n_points)
    ndim = out_dim
    if bas_length is None:
        bas_length = max(ndim, m)
    if bas_length > n:
        bas_length = n
    a_bas = np.copy(a[:bas_length])
    a_bas_base = np.copy(a_bas)
    c = np.dot(a, la.pinv(a_bas))
    block_indcs = np.arange(int(n // ndim))
    uniq_base_idx = np.copy(block_indcs[:(int(bas_length // ndim))])
    weights = np.ones(bas_length // ndim)
    s = cold_start_tens(c, ndim)

    for i in range(n_iter):
        det_list = la.det(np.eye(ndim) + s)
        elem = np.argmax(det_list)
        if elem in uniq_base_idx:
            ix = np.where(uniq_base_idx == elem)[0][0]
            weights[ix] += 1
            if to_use_recalc:
                c1 = np.copy(c)
                c = proj_matrix_upd(c1, ndim, ix, np.sqrt(weights[ix] / (weights[ix] - 1)))
            else:
                range_new_block = np.arange(ix * ndim, ix * ndim + ndim)
                a_bas[range_new_block, :] = np.sqrt(weights[ix]) * a_bas_base[range_new_block, :]
                c = np.dot(a, la.pinv(a_bas))
        else:
            range_new_block = np.arange(elem * ndim, elem * ndim + ndim)
            a_bas = np.vstack((a_bas, a[range_new_block]))
            a_bas_base = np.vstack((a_bas_base, a[range_new_block]))
            uniq_base_idx = np.append(uniq_base_idx, block_indcs[elem])
            weights = np.append(weights, np.array(1))
            c = np.dot(a, la.pinv(a_bas))
        s = cold_start_tens(c, ndim)

    return pivs[uniq_base_idx], weights.astype(int)


@jit
def cold_start_tens(c, ndim):
    m = c.shape[1]
    num_block = c.shape[0] // ndim
    if ndim < c.shape[1]:
        S = np.empty((num_block, ndim, ndim))
        for i in range(num_block):
            S[i, :, :] = np.dot(c[i * ndim:i * ndim + ndim], c[i * ndim:i * ndim + ndim].T)
    else:
        S = np.empty((num_block, m, m))
        for i in range(num_block):
            S[i, :, :] = np.dot(c[i * ndim:i * ndim + ndim].T, c[i * ndim:i * ndim + ndim])
    return S


def maxvol_precondition(a, pivs):
    k = a.shape[1]
    indc, _ = rect_maxvol(a, maxK=max_pts(k))
    idx_n, idx_o = change_intersept(np.arange(indc.shape[0]), indc)
    a[idx_n, :] = a[idx_o, :]
    pivs[idx_n] = pivs[idx_o]
    return a, pivs


def max_pts(m):
    return int(m * (m + 1) / 2)


