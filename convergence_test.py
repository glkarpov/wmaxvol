import getopt
import pathlib
import os
from exp_setup import *

# For Noisy and Hachiko computations
# os.environ['OMP_NUM_THREADS'] = '6'
# print(os.environ['OMP_NUM_THREADS'])


to_print_progress = False


def debug_print(s):
    if to_print_progress:
        print(s)
        sys.stdout.flush()


def convergence_plotter(error_matrix, use_log=False):
    n_iters = error_matrix.shape[1]
    z = 1.96  # z-value for 95% confidence interval
    k = z / np.sqrt(error_matrix.shape[0])
    mean_tensor = np.mean(error_matrix, axis=0)
    std = np.std(error_matrix, axis=0)
    var_up = mean_tensor + k * std
    var_down = mean_tensor - k * std
    fig = plt.figure()
    if use_log:
        plt.yscale('log')
    plt.plot(np.arange(n_iters), mean_tensor, linewidth=0.8)
    plt.fill_between(np.arange(n_iters), var_down, var_up, alpha=0.8, label='95% CI_')
    plt.grid(True)
    plt.show()


def parallel_sim(a, out_dim, n_parts):
    n, m = a.shape
    n_iter = 1600
    k = int(n / n_parts)
    n_local_points = int(k / out_dim)
    n_global_points = int(n / out_dim)
    block_global_indcs = np.arange(n_global_points)
    result = []
    for i in range(n_parts):
        full_piv_cur_range = np.arange(k * i, k * (i + 1))
        points_cur_range = block_global_indcs[n_local_points * i: n_local_points * (i + 1)]
        pts_i, wts_i = wmaxvol(a[full_piv_cur_range], n_iter=n_iter, out_dim=out_dim, do_mv_precondition=False)
        print(pts_i, 'pts')
        for j, elem in enumerate(pts_i):
            result.append(points_cur_range[elem])
    # print(np.array(result))
    # print(len(result))
    block_trust_indc = np.zeros(out_dim * len(result), dtype=int)
    for idx, elem in enumerate(result):
        block_trust_indc[idx * out_dim: out_dim * (idx + 1)] = np.arange(elem * out_dim, (elem + 1) * out_dim)
    a_new = np.copy(a[block_trust_indc, :])
    pts, wts = ExperimentRun.wmaxvol_search(a_new, n_iter, out_dim)
    print(wts, 'wts finale')
    pts = np.array(pts, dtype=int)
    result = np.array(result, dtype = int)
    finale = result[pts]
    print(np.sort(finale), 'finale')


def parallel_outer_stat_test():
    design_space_cardinality = 120
    out_dim = 1
    design_dimension = 1
    expansion = 5
    derivative = False
    n_parts = 4
    x = complex_area_pnts_gen(design_space_cardinality, design_dimension, distrib='lhs',
                              mod=None)
    a = GenMat(expansion * out_dim, x, poly=cheb, debug=False, pow_p=1,
               ToGenDiff=derivative)
    if derivative:
        a = matrix_prep(a, out_dim)
    print(a.shape, 'matrix shape')

    strforward, wts_str = ExperimentRun.wmaxvol_search(a, 1200, out_dim)

    parallel_sim(a, out_dim, n_parts)
    print(np.sort(strforward), 'straight')
    print(wts_str, 'wts_straight')


def block_dim_finder(design_dim_arr, mult_arr):
    combs = list(itertools.product(design_dim_arr, mult_arr))
    for (i, j) in combs:
        sm = 0
        for random_op in range(6):
            x = complex_area_pnts_gen(100, i, distrib='lhs', mod=None)
            a = GenMat(j * (i + 1), x, poly=cheb)
            a = matrix_prep(a, i + 1)
            if np.linalg.matrix_rank(a[:a.shape[1]]) == a.shape[1]:
                sm += 1
        if sm == 6:
            print(i, j)
    print('Done')


def main():
    cur_pos = str(pathlib.Path(__file__).parent.absolute())
    # cur_pos = "/trinity/home/g.karpov/maxvol-approximation" # For Zhores
    config = Config()
    global_iters = 1
    try:
        opts, args = getopt.getopt(sys.argv[1:], 'a:b:c:d:e:f:',
                                   ['global_iters=', 'wmxvl_iters=', 'ndim=', 'npts=', 'maxex=', 'add_name='])
        for currentArgument, currentValue in opts:
            if currentArgument in ("-a", "--global_iters"):
                global_iters = int(currentValue)
            if currentArgument in ("-b", "--wmxvl_iters"):
                config.n_iter = int(currentValue)
            elif currentArgument in ("-c", "--ndim"):
                config.design_dimension = int(currentValue)
            elif currentArgument in ("-d", "--npts"):
                config.design_space_cardinality = int(currentValue)
            elif currentArgument in ("-e", "--maxex"):
                config.max_expansion = int(currentValue)
            elif currentArgument in ("-f", "--add_name"):
                adder_name = currentValue
    except getopt.GetoptError:
        print('Parsing error')
        sys.exit(2)
    debug_print(
        "Experiment configured with parameters: global iters = {}, local iters = {}, npts = {}, ndim = {}, ex = {}".format(
            global_iters, config.n_iter, config.design_space_cardinality, config.design_dimension,
            config.max_expansion))
    config.out_dim = config.design_dimension + 1
    config.derivative = True
    eps_matrix = np.empty((global_iters, config.n_iter))
    frac_matrix = np.empty((global_iters, config.n_iter))

    dir_str = cur_pos + '/convergence_test_dim={}'.format(config.design_dimension)

    try:
        os.makedirs(dir_str)
    except:
        pass
    exp_name = 'p={}_pts={}_{}'.format(config.max_expansion * config.out_dim, config.design_space_cardinality, adder_name)

    for i_global in range(global_iters):
        x = complex_area_pnts_gen(config.design_space_cardinality, config.design_dimension, distrib='lhs',
                                  mod=config.domain_type)
        a = GenMat(config.max_expansion * config.out_dim, x, poly=config.poly, debug=False, pow_p=config.pow_p,
                   ToGenDiff=config.derivative)
        if config.derivative:
            a = matrix_prep(a, config.out_dim)
        debug_print("Matrix generated on iteration {}, with shape {}".format(i_global, a.shape))
        debug_print("Rank of upper square submatrix = {}".format(np.linalg.matrix_rank(a[:a.shape[1]])))
        _, wts, eps_i, frac_i = wmaxvol_analysis(a, config.n_iter, config.out_dim, filtration=True)
        eps_matrix[i_global, :] = eps_i
        frac_matrix[i_global, :] = frac_i
        debug_print("Analysis done on iteration {}".format(i_global))

    np.savez(os.path.join(dir_str, exp_name), eps=eps_matrix, frac=frac_matrix)


if __name__ == "__main__":
    main()

    # cur_pos = str(pathlib.Path(__file__).parent.absolute())
    # dir_str = cur_pos + '/convergence_test_dim=1/'
    # exp_name = 'p=12_pts=200.npz'
    # data = np.load(dir_str + exp_name)
    # eps_matrix = data['eps']
    # frac_matrix = data['frac']
    # print(frac_matrix[-10:, -10:], frac_matrix.shape)
    # convergence_plotter(eps_matrix, use_log=False)
    # convergence_plotter(frac_matrix, use_log=False)

    #parallel_outer_stat_test()
