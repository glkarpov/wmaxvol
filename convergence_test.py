import getopt
import pathlib
import os
from exp_setup import *

os.environ['OMP_NUM_THREADS'] = '6'
print(os.environ['OMP_NUM_THREADS'])


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
    plt.fill_between(np.arange(n_iters), var_down, var_up, alpha=0.4, label='95% CI_')
    plt.grid(True)
    plt.show()


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
    config = Config()
    global_iters = 1
    try:
        opts, args = getopt.getopt(sys.argv[1:], 'a:b:c:d:e:',
                                   ['global_iters=', 'wmxvl_iters=', 'ndim=', 'npts=', 'maxex='])
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
    except getopt.GetoptError:
        print('Parsing error')
        sys.exit(2)
    config.out_dim = config.design_dimension + 1
    config.derivative = True
    eps_matrix = np.empty((global_iters, config.n_iter))
    frac_matrix = np.empty((global_iters, config.n_iter))

    dir_str = cur_pos + '/convergence_test'
    try:
        os.makedirs(dir_str)
    except:
        pass
    exp_name = 'dim={}_p={}'.format(config.design_dimension, config.max_expansion * config.out_dim)

    for i_global in range(global_iters):
        x = complex_area_pnts_gen(config.design_space_cardinality, config.design_dimension, distrib='lhs',
                                  mod=config.domain_type)
        a = GenMat(config.max_expansion * config.out_dim, x, poly=config.poly, debug=False, pow_p=config.pow_p,
                   ToGenDiff=config.derivative)
        if config.derivative:
            a = matrix_prep(a, config.out_dim)
        print(np.linalg.matrix_rank(a[:a.shape[1]]))
        _, wts, eps_i, frac_i = wmaxvol_analysis(a, config.n_iter, config.out_dim, filtration=True)
        eps_matrix[i_global, :] = eps_i
        frac_matrix[i_global, :] = frac_i

    np.savez(os.path.join(dir_str, exp_name), eps=eps_matrix, frac=frac_matrix)


if __name__ == "__main__":
    main()
