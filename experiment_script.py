import getopt

from mva_test import *
import os
os.environ['OMP_NUM_THREADS'] = '6'
print(os.environ['OMP_NUM_THREADS'])


def main():
    global min_expansion, max_expansion, max_row
    domain_type = None
    nder = 2
    num_points_for_big_matrix = 200
    n_test = 5000  # points on test grid (for calculating error on final step)
    cut_radius = 0.005
    try:
        opts, args = getopt.getopt(sys.argv[1:], 'a:b:c:d:e:', ['minex=', 'maxex=', 'maxpts=', 'domtype=', 'cutrad='])
        for currentArgument, currentValue in opts:
            if currentArgument in ("-a", "--minex"):
                min_expansion = int(currentValue)
            elif currentArgument in ("-b", "--maxex"):
                max_expansion = int(currentValue)
            elif currentArgument in ("-c", "--maxpts"):
                max_row = int(currentValue)
            elif currentArgument in ("-d", "--domtype"):
                domain_type = None if currentValue == 'None' else currentValue
            elif currentArgument in ("-e", "--cutrad"):
                cut_radius = float(currentValue)
    except getopt.GetoptError:
        print('Parsing error')
        sys.exit(2)

    initial_points_distrib = 'LHS'

    basis_func_type = cheb  # used polynomials
    add_str = '-'.join([str(i) for i in [max_expansion, max_row, (domain_type if domain_type else 'Square')]])
    dir_str = './cr_test_' + add_str

    dir_pdf = os.path.join(dir_str, "pdf")
    try:
        os.makedirs(dir_pdf)
    except:
        pass

    # ---------------------------------
    p_size = (nder + 1) * max_row  # number of rows in big matrix

    # evaluating test
    points_fn = 'taken_points_{}'.format(initial_points_distrib)
    try:
        taken_points = np.load(os.path.join(dir_str, points_fn) + ".npz")
        design_space = taken_points['design_space']
    except:
        design_space = complex_area_pnts_gen(num_points_for_big_matrix, nder, distrib='lhs', mod=domain_type)
        points_test = complex_area_pnts_gen(n_test, nder, distrib=initial_points_distrib, mod=domain_type)
        np.savez(os.path.join(dir_str, points_fn), x=design_space, points_test=points_test)

    A = GenMat(p_size, design_space, poly=basis_func_type, debug=False, pow_p=1)
    A = matrix_prep(A, nder + 1)

    fn_pre_pdf = "distrib={}".format(initial_points_distrib)

    # f = open(os.path.join(dir_str, "distrib={}_radius={}".format(initial_points_distrib, cut_radius) + '.txt'), "w")
    f = open(os.path.join(dir_str, "distrib={}".format(initial_points_distrib) + '.txt'), "w")
    for expansion in range(min_expansion, max_expansion + 1):
        for N_rows_ex in range(max_row, expansion, -1):  # It's not the way people do...
            N_rows = N_rows_ex * (nder + 1)
            fnpdf = os.path.join(dir_pdf, fn_pre_pdf + "_expansion={}_N_rows_ex={}.pdf".format(expansion, N_rows_ex))
            try:
                taken_points = test_bm(A, design_space, nder, expansion, N_rows, to_save_pivs=N_rows_ex == max_row,
                                       fnpdf=fnpdf)
            except SingularError as err:
                print('not full column rank with expansion={}, N_rows_ex={}, err={}'.format(
                    expansion, N_rows_ex, err.value))
                # continue
                break

            taken_points.tofile(f, sep=" ")
            f.write("_Nrows={}_expans={}\n".format(N_rows, expansion))
            f.flush()

    f.close()


if __name__ == "__main__":
    main()
