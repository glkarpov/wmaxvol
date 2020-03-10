import pathlib
from exp_setup import *
from exp_proccess import *

os.environ['OMP_NUM_THREADS'] = '6'
print(os.environ['OMP_NUM_THREADS'])


def main():
    cur_pos = str(pathlib.Path(__file__).parent.absolute())
    ndim = 2
    k_first = 12
    exp_folder = "/domain_exp_35-_lebesgue_dim{}/".format(ndim)
    dir_points = cur_pos + exp_folder
    design_space = "domain_dim={}".format(ndim)
    calc_design = "designs_dim={}.txt".format(ndim)

    taken_points = np.load(dir_points + design_space + ".npz")
    x = taken_points['x']
    ex = experiment(dir_points + design_space, dir_points + calc_design, ndim, 1)

    expansions = np.unique(ex.cardinalities)[:k_first]
    config = Config()
    config.expansion_set = expansions
    config.n_iter = expansions[-1] * 1000


if __name__ == "__main__":
    main()
