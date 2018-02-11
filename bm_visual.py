import numpy as np
import sys
import matplotlib.pyplot as plt
import matplotlib
from mpl_toolkits.mplot3d import Axes3D
import re
from matplotlib import cm


def file_extraction(Filepath):
    srch = re.compile(r'([\d\s]+)_error=([\+\-\d\.eE]+)_Nrows=(\d+)_expans=(\d+)')
    fnd = srch.findall(open(Filepath, 'r').read())
    return tuple(np.array(i) for i in zip(*[(float(i0), int(i1), int(i2), [int(p) for p in im1.strip().split(' ') if len(p) > 0])
                                            for im1, i0, i1, i2 in fnd]))


def DataToMesh(error, N_row, N_col, *args):
    row_s = sorted(list(set(N_row)))
    col_s = sorted(list(set(N_col)))
    data = {(N_row[i], N_col[i]) : e for i, e in enumerate(error)}
    
    res = np.empty((len(row_s), len(col_s)), dtype=float)
    for i, r in enumerate(row_s):
        for j, c in enumerate(col_s):
            try:
                res[i,j] = data[(r, c)]
            except:
                res[i,j] = np.nan
    X, Y = np.meshgrid(row_s, col_s)
    return res.T, X, Y


def PlotError(fn, log_it=False):
    error, N_row, N_col = DataToMesh(*file_extraction(fn))
    if log_it:
        error = np.log10(error)

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_surface(N_row, N_col, error, edgecolor='black', linewidth=0.5, cmap = cm.Spectral)
    # ax.legend()
    ax.set_xlabel('N_rows', fontsize=10)
    ax.set_ylabel('N_columns')
    plt.show()


def PlotPoints(fn, row, col, x, to_save_fig=False, fn_out="func.pdf"):
    _, N_row, N_col, idx = file_extraction(fn)

    l_bound = np.amin(x, 0)
    u_bound = np.amax(x, 0)
    delta = (u_bound - l_bound)/20.0
    plt.xlim(l_bound[0] - delta[0], u_bound[0] + delta[0])
    plt.ylim(l_bound[1] - delta[1], u_bound[1] + delta[1])
    plt.plot(x[idx, 0], x[idx, 1], 'b^')
    # plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=2, borderaxespad=0.)
    # plt.title("E = {}".format(error))
    plt.grid(True)
    if to_save_fig:
        # fn = 'func={}_d={}_num={}_nder={}.pdf'.format(function.__name__, N_column, N_rows, nder)
        plt.savefig(fn_out)




if __name__ == '__main__':
    if len(sys.argv) > 1:
        fn = sys.argv[1]
    else:
        fn = "func=Rosenbrock_poly=cheb.txt"

    PlotError(fn, log_it=False)


