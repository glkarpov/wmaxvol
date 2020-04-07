import getopt
import pathlib
from exp_proccess import *
import os

os.environ['OMP_NUM_THREADS'] = '6'
print(os.environ['OMP_NUM_THREADS'])


def main():
    N_iter = 50
    test_space_cardinality = 50000
    try:
        opts, args = getopt.getopt(sys.argv[1:], 'a:b:c:', ['ndim=', 'experiment=', 'func='])
        for currentArgument, currentValue in opts:
            if currentArgument in ("-a", "--ndim"):
                ndim = int(currentValue)
            elif currentArgument in ("-b", "--experiment"):
                exp_name = str(currentValue)
                calc_design = str(currentValue)+".txt"
            elif currentArgument in ("-c", "--func"):
                function = None if currentValue == 'None' else eval(currentValue)
    except getopt.GetoptError:
        print('Parsing error')
        sys.exit(2)

    cur_pos = str(pathlib.Path(__file__).parent.absolute())
    exp_folder = "/domain_exp_70-_lebesgue_dim{}/".format(ndim)
    dir_points = cur_pos + exp_folder
    design_space = "domain_dim={}".format(ndim)
    test_design_space = "test_domain_dim={}".format(ndim)
    ex = experiment(dir_points + design_space, dir_points + calc_design, ndim, 1)
    try:
        domain = np.load(dir_points + test_design_space + ".npz")
        points_test = domain['x']
    except:
        points_test = complex_area_pnts_gen(test_space_cardinality, ndim, distrib='lhs',
                                  mod=None)
        np.savez(os.path.join(dir_points, test_design_space), x=points_test)

    ### Experiment setup
    error_set = ['random','lhs','sobol']
    col_to_fix = []
    point_to_fix = ["all"]
    slice_coeff = None
    exp_solve = [col_to_fix, point_to_fix, slice_coeff]
    mask = index_preprocess(ex.p_indices, ex.expans, exp_solve)

    ## Error calculation
    if type(function) is not list:
        function = [function]
    mesh = np.copy(mask)
    mesh[:, 1] = ex.expans[mask[:, 1]]
    # for function in function_set:
    error_tensor = mult_error_tensor(N_iter, mesh, function, points_test, error_set, shape=None)

    if function == [None]:
        tensor_name = 'error_leb_'
    else:
        tensor_name = 'error_{}_'.format(function[0].__name__)
    np.savez(os.path.join(dir_points, tensor_name)+exp_name, error_tensor=error_tensor)


if __name__ == "__main__":
    main()
