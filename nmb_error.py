import getopt
import pathlib

from exp_proccess import *

os.environ['OMP_NUM_THREADS'] = '6'
print(os.environ['OMP_NUM_THREADS'])


def process_experiment_scheme(scheme, error_types, block_size, min_exp, max_exp, max_num_of_pts):
    storage = {}
    for column_expansion in scheme[0]:
        for n_row in range((column_expansion + 1) * block_size, (max_num_of_pts + 1) * block_size):
            storage[(n_row, (column_expansion * block_size))] = dict()
            for error_type in error_types:
                storage[(n_row, (column_expansion * block_size))][error_type] = list()

    for num_of_pts in scheme[1]:
        for n_col in range(min_exp * block_size, (max_exp + 1) * block_size):
            storage[(num_of_pts * block_size, n_col)] = dict()
            for error_type in error_types:
                storage[(num_of_pts * block_size, n_col)][error_type] = list()

    for slice_coefficient in scheme[2]:
        for column_expansion in range(min_exp, max_exp + 1):
            num_of_pts = int(slice_coefficient * column_expansion)
            if num_of_pts <= max_num_of_pts:
                storage[(num_of_pts * block_size, column_expansion * block_size)] = dict()
                for error_type in error_types:
                    storage[(num_of_pts * block_size, column_expansion * block_size)][error_type] = list()

    return storage


def model_error_calculation(n_iter, function, points_test, error_dict, poly_basis, pow_poly=1, shape=None,
                            derivative=False):
    dim = points_test.shape[1]

    if type(function) is not list:
        function = [function]
    if function == [None]:
        ValsandNorms = None
    else:
        ValsandNorms = MakeValsAndNorms(function, points_test)
    for s in np.arange(n_iter):
        print('Iteration #{}'.format(s))
        for model_matrix_size in error_dict:
            n_row, n_col = model_matrix_size[0], model_matrix_size[1]
            n_pts = int(n_row / dim)
            error_at_sampling = error_dict[model_matrix_size]
            for sampling_type in error_at_sampling:
                x_tmp = complex_area_pnts_gen(n_pts, dim, mod=shape, distrib=sampling_type)
                if sampling_type == 'sobol' and len((np.where(x_tmp == np.array([[0., 0.], ])))[0]) != 0:

                    while True:
                        x_tmp = complex_area_pnts_gen(n_pts, dim, mod=shape, distrib='sobol')
                        if len((np.where(x_tmp == np.array([[0., 0.], ])))[0]) == 0:
                            break
                if function == [None]:
                    error = LebesgueConst(x_tmp, n_col, poly=poly_basis, test_pnts=points_test,
                                          pow_p=pow_poly, funcs=ValsandNorms, derivative=derivative)
                else:
                    _, error = LebesgueConst(x_tmp, n_col, poly=poly_basis, test_pnts=points_test,
                                             pow_p=pow_poly, funcs=ValsandNorms, derivative=derivative)
                error_at_sampling[sampling_type].append(error)

    return True


def extracted_design_to_dict(row_array, col_array, idx_array, block_size):
    design_dict = dict()
    assert (row_array.shape[0] == col_array.shape[0])

    for i in range(row_array.shape[0]):
        design_dict[(row_array[i], col_array[i] * block_size)] = idx_array[i]

    return design_dict


def bmaxvol_error(function, points_test, design_domain, design_dict, error_dict, poly_basis, pow_poly=1):
    b_error = dict()
    if type(function) is not list:
        function = [function]

    ValsandNorms = MakeValsAndNorms(function, points_test)
    for model_matrix_size in error_dict:
        n_row, n_col = model_matrix_size[0], model_matrix_size[1]
        try:
            design_idx = design_dict[(n_row, n_col)]
            if function == [None]:
                error = LebesgueConst(design_domain[design_idx], n_col, poly=poly_basis, test_pnts=points_test,
                                      pow_p=pow_poly, funcs=ValsandNorms, derivative=True)
            else:
                _, error = LebesgueConst(design_domain[design_idx], n_col, poly=poly_basis, test_pnts=points_test,
                                         pow_p=pow_poly, funcs=ValsandNorms, derivative=True)

            b_error[(n_row, n_col)] = error
        except:
            continue
    return b_error


def process_maxvol_error_for_plot(error_dict, block_size, min_exp, max_exp, max_num_of_pts, mode=()):
    exp_type, parameter = mode[0], mode[1]
    x_axis, error_value = [], []
    if exp_type == "fixed_num_of_functions":
        n_col = parameter * block_size
        for n_row in range((parameter + 1) * block_size, (max_num_of_pts + 1) * block_size, block_size):
            try:
                error_at_model_size = error_dict[(n_row, n_col)]
                error_value.append(error_at_model_size)
                x_axis.append(int(n_row / block_size))
            except:
                continue

    if exp_type == "fixed_num_of_points":
        n_row = parameter * block_size
        for n_col in range(min_exp * block_size, (max_exp + 1) * block_size, block_size):
            try:
                error_at_model_size = error_dict[(n_row, n_col)]
                error_value.append(error_at_model_size)
                x_axis.append(n_col)
            except:
                continue

    if exp_type == "slice":
        for column_expansion in range(min_exp, max_exp + 1):
            num_of_pts = int(parameter * column_expansion)
            n_row = num_of_pts * block_size
            try:
                error_at_model_size = error_dict[(n_row, column_expansion * block_size)]
                error_value.append(error_at_model_size)
                x_axis.append(int(n_row / block_size))
            except:
                continue
    return np.array(x_axis), np.array(error_value)


def process_error_for_plot(sampling_types, error_dict, block_size, min_exp, max_exp, max_num_of_pts, mode=()):
    data_for_plot = {}
    z_value = 1.28
    for sampling_type in sampling_types:
        data_for_plot[sampling_type] = ([], [], [])
    exp_type, parameter = mode[0], mode[1]

    if exp_type == "fixed_num_of_functions":
        n_col = parameter * block_size
        for n_row in range((parameter + 1) * block_size, (max_num_of_pts + 1) * block_size, block_size):
            error_at_model_size = error_dict[(n_row, n_col)]
            for sampling_type in sampling_types:
                data_for_plot[sampling_type][0].append(np.mean(error_at_model_size[sampling_type]))
                data_for_plot[sampling_type][1].append(np.std(error_at_model_size[sampling_type], ddof=1))
                data_for_plot[sampling_type][2].append(int(n_row / block_size))

    if exp_type == "fixed_num_of_points":
        n_row = parameter * block_size
        for n_col in range(min_exp * block_size, (max_exp + 1) * block_size, block_size):
            error_at_model_size = error_dict[(n_row, n_col)]
            for sampling_type in sampling_types:
                data_for_plot[sampling_type][0].append(np.mean(error_at_model_size[sampling_type]))
                data_for_plot[sampling_type][1].append(np.std(error_at_model_size[sampling_type], ddof=1))
                data_for_plot[sampling_type][2].append(n_col)

    if exp_type == "slice":
        for column_expansion in range(min_exp, max_exp + 1):
            num_of_pts = int(parameter * column_expansion)
            if num_of_pts <= max_num_of_pts:
                n_row = num_of_pts * block_size
                error_at_model_size = error_dict[(n_row, column_expansion * block_size)]
                for sampling_type in sampling_types:
                    data_for_plot[sampling_type][0].append(np.mean(error_at_model_size[sampling_type]))
                    data_for_plot[sampling_type][1].append(np.std(error_at_model_size[sampling_type], ddof=1))
                    data_for_plot[sampling_type][2].append(int(n_row / block_size))

    for sampling_type in sampling_types:
        mean_array = np.array(data_for_plot[sampling_type][0])
        x_axis = np.array(data_for_plot[sampling_type][2])
        upper_bound = mean_array + np.array(data_for_plot[sampling_type][1]) * (z_value / np.sqrt(mean_array.shape[0]))
        lower_bound = mean_array - np.array(data_for_plot[sampling_type][1]) * (z_value / np.sqrt(mean_array.shape[0]))
        data_for_plot[sampling_type] = (x_axis, mean_array, lower_bound, upper_bound)
    return data_for_plot


def main():
    N_iter = 4
    col_to_fix = []
    point_to_fix = []
    slice_coeff = []
    try:
        opts, args = getopt.getopt(sys.argv[1:], 'a:b:c:d:e:',
                                   ['add_param=', 'func=', 'col_fix=', 'pts_fix=', 'slice='])
        for currentArgument, currentValue in opts:
            if currentArgument in ("-a", "--add_param"):
                add_name = None if currentValue == 'None' else '_' + str(currentValue)
            elif currentArgument in ("-b", "--func"):
                function = None if currentValue == 'None' else eval(currentValue)
            elif currentArgument in ("-c", "--col_fix"):
                col_to_fix = [] if currentValue == 'None' else [int(v) for v in currentValue.split(',')]
            elif currentArgument in ("-d", "--pts_fix"):
                point_to_fix = [] if currentValue == 'None' else [int(v) for v in currentValue.split(',')]
            elif currentArgument in ("-e", "--slice"):
                slice_coeff = [] if currentValue == 'None' else [float(v) for v in currentValue.split(',')]

    except getopt.GetoptError:
        print('Parsing error')
        sys.exit(2)

    cur_pos = str(pathlib.Path(__file__).parent.absolute())
    exp_name = "doe_exp"
    exp_dir = os.path.join(cur_pos, exp_name + add_name)

    calc_design = 'distrib=LHS.txt'
    pts_filename = 'taken_points_LHS.npz'

    path_calculated_design = os.path.join(exp_dir, calc_design)
    path_generated_pts = os.path.join(exp_dir, pts_filename)

    domain = np.load(path_generated_pts)
    points_test = domain['points_test']
    design_space = domain['x']
    n_dim = design_space.shape[1]

    n_row, n_col, pts_indices = file_extraction(path_calculated_design)
    min_exp, max_exp = np.min(n_col), np.max(n_col)
    max_pts = int(np.max(n_row) / (n_dim + 1))
    design_dict = extracted_design_to_dict(n_row, n_col, pts_indices, n_dim + 1)
    # pow_p = 1
    poly_type = cheb

    # Experiment setup
    error_blank = ('lhs', 'sobol', 'random')

    if len(col_to_fix) + len(point_to_fix) + len(slice_coeff) == 0:
        exp_solve = [np.arange(min_exp, max_exp + 1), [], []]
    else:
        exp_solve = [col_to_fix, point_to_fix, slice_coeff]
        print(exp_solve)

    error_storage_dict = process_experiment_scheme(exp_solve, error_blank, n_dim + 1, min_exp, max_exp, max_pts)
    model_error_calculation(N_iter, function, points_test, error_storage_dict, poly_basis=cheb, derivative=True)
    bmaxvol_stat = bmaxvol_error(function, points_test, design_space, design_dict, error_storage_dict,
                                 poly_basis=poly_type)

    if function is None:
        tensor_name = 'error_leb'
    else:
        tensor_name = 'error_{}'.format(function.__name__)
    np.savez(os.path.join(exp_dir, tensor_name), nmb_error_dict=error_storage_dict, maxvol_error_dict=bmaxvol_stat)


if __name__ == "__main__":
    main()
