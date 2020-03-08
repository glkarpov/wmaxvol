import numpy as np
from gen_mat import *
from wmaxvol import *
from gen_points import *
from mva_test import *
import os


class Config:
    def __init__(self):
        self.min_expansion = None
        self.max_expansion = None
        self.design_space = None
        self.model_matrix = None
        self.design_dimension = None
        self.domain_type = None
        self.out_dim = 1
        self.derivative = False
        self.n_iter = 500
        self.delta_n = 0
        self.design_space_cardinality = 2000
        self.poly = cheb
        self.to_apply_mask = False
        self.mask = None

    def load_external_model(self, model_matrix, out_dim):
        self.model_matrix = model_matrix
        self.min_expansion = model_matrix.shape[1]
        self.max_expansion = self.min_expansion + 1
        self.out_dim = out_dim

    def load_external_space(self, d, poly, out_dim):
        self.design_space = d
        self.design_space_cardinality = d.shape[0]
        self.design_dimension = d.shape[1]
        self.out_dim = out_dim
        self.derivative = True
        self.poly = poly

    def mask_apply(self, x):
        dim = self.design_dimension
        for i in range(dim):
            delta = self.mask[i, 1] - self.mask[i, 0]
            x[:, i] = (delta / 2) * x[:, i] + self.mask[i, 0]
        return x


class Experiment_run:
    def __init__(self, config, results_folder):
        self.config = config
        self.results_folder = results_folder
        try:
            os.makedirs(self.results_folder)
        except:
            pass
        self.domain_fn = 'domain_' + 'dim = {}'.format(self.config.design_dimension)

    @staticmethod
    def wmaxvol_search(A, n_iter, out_dim):
        pts, wts = wmaxvol(A, n_iter=n_iter, out_dim=out_dim, do_mv_precondition=True)
        k = A.shape[1]
        fo_border = n_iter / (2 * max_pts(k))
        wts_trust = np.where(wts > fo_border)[0]
        ps = pts[wts_trust]
        return ps, wts[wts_trust].astype(int)

    def run(self):
        setup = self.config
        if setup.model_matrix is None:
            x = complex_area_pnts_gen(setup.design_space_cardinality, setup.design_dimension, distrib='lhs',
                                      mod=setup.domain_type)
            if setup.to_apply_mask:
                x = setup.mask_apply(x)
            np.savez(os.path.join(self.results_folder, self.domain_fn), x=x)
            m = GenMat(setup.max_expansion * setup.out_dim, x, poly=setup.poly, debug=False, pow_p=1,
                       ToGenDiff=setup.derivative)
            if setup.derivative:
                m = matrix_prep(m, ndim)
        else:
            m = setup.model_matrix

        f = open(os.path.join(self.results_folder, "designs_dim={}".format(setup.design_dimension) + '.txt'), "w")
        for expansion in range(setup.min_expansion, setup.max_expansion):
            a = np.copy(m[:, :expansion * setup.out_dim])
            try:
                des_points, weights = self.wmaxvol_search(a, setup.n_iter, setup.out_dim)
            except SingularError as err:
                print('not full column rank with expansion={}'.format(
                    expansion))
                # continue
                break
            des_points.tofile(f, sep=" ")
            f.write("_expans={}\n".format(expansion))
            weights.tofile(f, sep=" ")
            f.write("_iter={}\n".format(setup.n_iter))
            setup.n_iter += expansion * setup.delta_n
            f.flush()
