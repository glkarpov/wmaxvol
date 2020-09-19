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


def maxvol_precondition(a, pivs, block_size):
    k = a.shape[1]
    if block_size == 1:
        indc, _ = rect_maxvol(a, maxK=max_pts(k))
    else:
        indc = rect_block_maxvol(a, block_size, k, max_iters=100)
    idx_n, idx_o = change_intersept(np.arange(indc.shape[0]), indc)
    a[idx_n, :] = a[idx_o, :]
    pivs[idx_n] = pivs[idx_o]
    return a, pivs


def max_pts(m):
    return int(m * (m + 1) / 2)


def sensitivity_eval(M, mu):
    d = np.trace(np.linalg.pinv(M) @ mu)
    return d


def get_variance(a, m, out_dim):
    k = int(a.shape[0] // out_dim)
    variance = np.empty(k)
    for i in range(k):
        block_i = a[i * out_dim: (i + 1) * out_dim]
        variance[i] = sensitivity_eval(m, block_i.T @ block_i)
    return variance


def wmaxvol_analysis(a, n_iter, out_dim, filtration=True, bas_length=None, do_mv_precondition=True):
    n, m = a.shape
    n_points = int(n / out_dim)
    if do_mv_precondition:
        a, pivs = maxvol_precondition(a, np.arange(n), out_dim)
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
    block_indcs = np.arange(int(n_points))
    uniq_base_idx = np.copy(block_indcs[:(int(bas_length // ndim))])
    weights = np.ones(bas_length // ndim)
    s = cold_start_tens(c, ndim)
    eps = []
    n_upper = []
    filtered = []
    added = []
    start = np.random.randint(n_iter // 4, n_iter // 2)
    fl_start = False
    for i in range(n_iter):
        if filtration and i == start:
            weights_old = np.copy(weights)
            adder = np.random.randint(m * (m + 1), 2 * m * (m + 1))
            adder += start
            fl_start = True
        if filtration and fl_start and i == adder:
            wts_up_to = np.zeros(weights.shape[0], dtype=int)
            wts_up_to[:weights_old.shape[0]] = np.copy(weights_old)
            weights_diff = weights - wts_up_to
            candidates = np.where(weights_diff == 0)[0]
            if candidates.shape[0] != 0:
                aim = np.argmin(sensitivity_vec[uniq_base_idx][candidates])
                filtered.append(uniq_base_idx[candidates][aim])
                weights = np.delete(weights, candidates[aim])
                uniq_base_idx = np.delete(uniq_base_idx, candidates[aim])
                a_bas = np.delete(a_bas, np.arange(candidates[aim] * out_dim, (candidates[aim] + 1) * out_dim), 0)
                a_bas_base = np.delete(a_bas_base,
                                       np.arange(candidates[aim] * out_dim, (candidates[aim] + 1) * out_dim), 0)
                c = np.dot(a, la.pinv(a_bas))
                s = cold_start_tens(c, ndim)

            weights_old = np.copy(weights)
            adder += np.random.randint(m * (m + 1), 2 * m * (m + 1))

        # if filtration and i == (n_iter // 2):
        #     fo_border = i / (2 * max_pts(m))
        #     wts_trust = np.where(weights > fo_border)[0]
        #     weights = weights[wts_trust]
        #     uniq_base_idx = uniq_base_idx[wts_trust]
        #     block_trust_indc = np.zeros(out_dim * len(wts_trust), dtype=int)
        #     for idx, elem in enumerate(wts_trust):
        #         block_trust_indc[idx * out_dim: out_dim*(idx+1)] = np.arange(elem * out_dim, (elem + 1) * out_dim)
        #     a_bas = a_bas[block_trust_indc, :]
        #     a_bas_base = a_bas_base[block_trust_indc,:]

        det_list = la.det(np.eye(ndim) + s)
        elem = np.argmax(det_list)
        if elem in uniq_base_idx:
            ix = np.where(uniq_base_idx == elem)[0][0]
            weights[ix] += 1
            range_new_block = np.arange(ix * ndim, ix * ndim + ndim)
            a_bas[range_new_block, :] = np.sqrt(weights[ix]) * np.copy(a_bas_base[range_new_block, :])
            c = np.dot(a, la.pinv(a_bas))
        else:
            added.append(block_indcs[elem])
            range_new_block = np.arange(elem * ndim, elem * ndim + ndim)
            a_bas = np.vstack((a_bas, a[range_new_block]))
            a_bas_base = np.vstack((a_bas_base, a[range_new_block]))
            uniq_base_idx = np.append(uniq_base_idx, block_indcs[elem])
            weights = np.append(weights, np.array(1))
            c = np.dot(a, la.pinv(a_bas))
        s = cold_start_tens(c, ndim)

        current_inf_matrix = a_bas.T @ a_bas / np.sum(weights)
        sensitivity_vec = get_variance(a, current_inf_matrix, ndim)
        sens_max = np.max(sensitivity_vec[uniq_base_idx]) if np.max(sensitivity_vec[uniq_base_idx]) > m else m
        sens_min = np.min(sensitivity_vec[uniq_base_idx]) if np.min(sensitivity_vec[uniq_base_idx]) < m else m
        eps.append(np.abs(sens_max - sens_min))
        outer_design_space = np.setdiff1d(block_indcs, uniq_base_idx)
        error_idcs = np.where(sensitivity_vec[outer_design_space] >= m)[0]
        n_upper.append(float(error_idcs.shape[0] / n_points))
    # print(sensitivity_vec[uniq_base_idx], 'sensitivity on design')
    # print(uniq_base_idx)
    # print(filtered)
    # print(added)

    return pivs[uniq_base_idx], weights.astype(int), eps, n_upper
