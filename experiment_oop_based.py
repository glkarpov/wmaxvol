import getopt
import pathlib
from exp_setup import *

os.environ['OMP_NUM_THREADS'] = '6'
print(os.environ['OMP_NUM_THREADS'])


def main():
    config = Config()
    cur_pos = str(pathlib.Path(__file__).parent.absolute())

    try:
        opts, args = getopt.getopt(sys.argv[1:], 'a:b:c:d:e:', ['minex=', 'maxex=', 'ndim=', 'domtype=', 'npts='])
        for currentArgument, currentValue in opts:
            if currentArgument in ("-a", "--minex"):
                config.min_expansion = int(currentValue)
            elif currentArgument in ("-b", "--maxex"):
                config.max_expansion = int(currentValue)
            elif currentArgument in ("-c", "--ndim"):
                config.design_dimension = int(currentValue)
            elif currentArgument in ("-d", "--domtype"):
                config.domain_type = None if currentValue == 'None' else currentValue
            elif currentArgument in ("-e", "--ndim"):
                config.design_space_cardinality = int(currentValue)
    except getopt.GetoptError:
        print('Parsing error')
        sys.exit(2)

    if config.design_space_cardinality is None:
        config.design_space_cardinality = 10000
    config.n_iter = 300
    config.delta_n = 10
    config.add_name = ''
    config.pow_p = 1
    add_str = '-'.join([str(i) for i in [config.max_expansion, "_lebesgue_dim{}".format(config.design_dimension)]])
    dir_str = cur_pos + '/domain_exp_' + add_str

    worker = ExperimentRun(config, dir_str)
    worker.run()


if __name__ == "__main__":
    main()
