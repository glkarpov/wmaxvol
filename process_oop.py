import pathlib
from exp_setup import *
from exp_proccess import *
import getopt

os.environ['OMP_NUM_THREADS'] = '6'
print(os.environ['OMP_NUM_THREADS'])


def main():
    try:
        opts, args = getopt.getopt(sys.argv[1:], 'a:', ['ndim='])
        for currentArgument, currentValue in opts:
            if currentArgument in ("-a", "--ndim"):
                ndim = int(currentValue)
    except getopt.GetoptError:
        print('Parsing error')
        sys.exit(2)

    cur_pos = str(pathlib.Path(__file__).parent.absolute())
    k_first = 15
    exp_folder = "/domain_exp_70-_lebesgue_dim{}/".format(ndim)
    dir_points = cur_pos + exp_folder
    design_space = "domain_dim={}".format(ndim)
    calc_design = "designs_dim={}.txt".format(ndim)

    taken_points = np.load(dir_points + design_space + ".npz")
    x = taken_points['x']
    ex = experiment(dir_points + design_space, dir_points + calc_design, ndim, 1)

    expansions = np.unique(ex.cardinalities)[:k_first]
    config = Config()
    config.load_external_space(x, cheb, 1)
    config.expansion_set = expansions
    config.n_iter = 200
    config.delta_n = 10
    config.add_name = 'expand'
    worker = Experiment_run(config, dir_points)
    worker.run()

if __name__ == "__main__":
    main()
