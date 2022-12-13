from mva_test import *
import os


class Experiment:
    def __init__(self, path_dspace, path_calculated_design, design_dim, out_dim):
        self.mask = None
        self.design_space_path = path_dspace
        self.ndim = design_dim
        self.out_dim = out_dim
        self.pow_p = 1
        self.poly = cheb
        self.derivative = False
        extract_design = file_extraction(path_calculated_design)
        if len(extract_design) == 3:
            self.weight_type = "uniweighted"
            self.n_rows = extract_design[0]
            self.expans = extract_design[1]
            self.p_indices = extract_design[2]

        else:
            self.weight_type = "weighted"
            self.expans = extract_design[0]
            self.p_indices = extract_design[1]
            self.weights = extract_design[3]
            self.weight_iter = [sum(j) for i, j in enumerate(self.weights)]
            self.norm_weights = False
            self.cardinalities = [len(p) for p in self.p_indices]

    def export_to_file(self, folder_path, add_name):
        if add_name is not '':
            add_name = '_' + add_name
        f = open(os.path.join(folder_path, "designs_dim={}{}".format(self.ndim, add_name) + '.txt'), "w")
        for i, expansion in enumerate(self.expans):
            np.array(self.p_indices[i]).tofile(f, sep=" ")
            f.write("_expans={}\n".format(expansion))
            np.array(self.weights[i]).tofile(f, sep=" ")
            f.write("_iter={}\n".format(self.weight_iter[i]))
            f.flush()

    def normalize_weights(self):
        if self.weight_type == "weighted" and not self.norm_weights:
            for i, iter in enumerate(self.weight_iter):
                self.weights[i] /= iter
            self.norm_weights = True
        else:
            print("already normalized")
        return ()

    def show_designs(self):
        for i, exp in enumerate(self.expans):
            print(self.p_indices[i], self.weights[i], exp)

    def apply_exp_mask(self, exp_mask):
        self.mask = index_preprocess(self.p_indices, self.expans, exp_mask)

    def index_preproc(self, exp_mask):
        indx = []
        expans_pattern = exp_mask[0]
        for exp_indx, expsn in enumerate(expans_pattern):
            inx = np.where(self.expans == expsn)[0][0]
            N_r = len(self.p_indices[inx])
            indx += [np.array((N_r, inx))]
        indx = np.array(indx)
        self.mask = np.unique(indx, axis=0)

    def error_calculation(self, function, points_test):
        taken_points = np.load(self.design_space_path + ".npz")
        x = taken_points['x']
        error = maxvol_error(x, function, points_test, self.p_indices, self.expans, self.mask,
                             derivative=self.derivative)
        return error

    def set_weights_ones(self):
        for i, wght in enumerate(self.weights):
            card = len(wght)
            self.weights[i] = np.ones(card).tolist()

    def update_cardinalities(self):
        self.cardinalities = [len(p) for p in self.p_indices]

    def update_weight_iter(self):
        self.weight_iter = [sum(j) for i, j in enumerate(self.weights)]

    def get_k_points(self, k, mode):
        if mode == "weights_down":
            for i, p_ind in enumerate(self.p_indices):
                if len(p_ind) >= k:
                    Z = [x for _, x in sorted(zip(self.weights[i], p_ind))]
                    self.p_indices[i] = Z[-k:]
                    self.weights[i] = sorted(self.weights[i])[-k:]
        elif mode == "equal_to_columns":
            for i, p_ind in enumerate(self.p_indices):
                Z = [x for _, x in sorted(zip(self.weights[i], p_ind))]
                self.p_indices[i] = Z[-self.expans[i]:]
                self.weights[i] = sorted(self.weights[i])[-self.expans[i]:]
                self.weight_iter = [sum(j) for i, j in enumerate(self.weights)]
        elif mode == "None":
            for i, p_ind in enumerate(self.p_indices):
                self.p_indices[i] = p_ind[:k]
                self.weights[i] = self.weights[i][:k]

    def change_expans(self, mode):
        if mode == "up_to_points":
            for i, p_ind in enumerate(self.p_indices):
                self.expans[i] = len(p_ind)

    def lebesgue_mask(self, points_test):
        Leb_e = []
        taken_points = np.load(self.design_space_path + ".npz")
        x = taken_points['x']
        for i in range(self.mask.shape[0]):
            rows_exp, col_exp = self.mask[i, 0], self.expans[self.mask[i, 1]]
            p = self.p_indices[self.mask[i, 1]][:rows_exp]
            b = LebesgueConst(x[p], col_exp * self.out_dim, poly=cheb, test_pnts=points_test, pow_p=1, wts=None,
                              funcs=None, derivative=self.derivative)
            Leb_e.append(b)
        return np.array(Leb_e).reshape((1, len(Leb_e)))

    def lebesgue_all(self, points_test, function, wwts=True):
        if wwts:
            arg = self.weights
        else:
            arg = None
        error = []
        taken_points = np.load(self.design_space_path + ".npz")
        x = taken_points['x']
        if type(function) is not list:
            function = [function]
        if function == [None]:
            ValsandNorms = None
        else:
            ValsandNorms = MakeValsAndNorms(function, points_test)

        for i, col_exp in enumerate(self.expans):
            p = self.p_indices[i]
            if function == [None]:
                b = LebesgueConst(x[p], col_exp, poly=self.poly, test_pnts=points_test, pow_p=self.pow_p,
                                  funcs=ValsandNorms, derivative=self.derivative)
            else:
                _, b = LebesgueConst(x[p], col_exp, poly=self.poly, test_pnts=points_test, pow_p=self.pow_p,
                                     funcs=ValsandNorms, derivative=self.derivative)

            error.append(b)
        return np.array(error).reshape((1, len(error)))


def index_preprocess(p_indices, n_col, exp_mask):
    indx = []
    for exp_indx, expsn in enumerate(n_col):
        N_r = len(p_indices[exp_indx])
        if expsn in exp_mask[0]:
            a = np.arange(expsn + 1, N_r + 1)
            indx += [np.array((i, j)) for i, j in zip(a.astype(int), itertools.repeat(exp_indx))]
        if exp_mask[-1]:
            scaler = int(np.around(exp_mask[-1] * expsn))
            indx += ([np.array((scaler, exp_indx))])

    for p_indx, n_r in enumerate(exp_mask[1]):
        if type(n_r) is int:
            a = np.where(n_col < int(n_r))[0]
            indx += [np.array((i, j)) for i, j in zip(itertools.repeat(int(n_r)), a)]
        elif n_r == "all":
            indx += [np.array((len(p_indices[i]), i)) for i, j in enumerate(n_col)]
    un = np.unique(np.array(indx), axis=0)

    z = [x for _, x in sorted(zip(un[:, 1], un[:, 0]))]
    a = np.sort(un[:, 1])
    un[:, 0] = z
    un[:, 1] = a
    return un


def submask(mask, crt_mask):
    rt = [(mask == crt_mask[i, :]).all(axis=1).nonzero()[0][0] for i in range(crt_mask.shape[0])]
    return rt


@jit
def mult_error_tensor(N_iter, mask, function, points_test, error_set, poly_basis, pow_poly=1, shape=None,
                      derivative=False):
    ndim = points_test.shape[1]
    error_tensor = np.empty((len(error_set), mask.shape[0], N_iter))
    if type(function) is not list:
        function = [function]
    if function == [None]:
        ValsandNorms = None
    else:
        ValsandNorms = MakeValsAndNorms(function, points_test)
    for s in np.arange(N_iter):
        print('Iteration #{}'.format(s))
        for i in range(mask.shape[0]):
            rows_exp, col_exp = mask[i, 0], mask[i, 1]
            for k, points_type in enumerate(error_set):
                x_tmp = complex_area_pnts_gen(rows_exp, ndim, mod=shape, distrib=points_type)
                if points_type == 'sobol' and len((np.where(x_tmp == np.array([[0., 0.], ])))[0]) != 0:

                    while True:
                        x_tmp = complex_area_pnts_gen(rows_exp, ndim, mod=shape, distrib='sobol')
                        if len((np.where(x_tmp == np.array([[0., 0.], ])))[0]) == 0:
                            break
                if function == [None]:
                    error_tensor[k][i, s] = LebesgueConst(x_tmp, col_exp * (ndim + 1), poly=poly_basis, test_pnts=points_test,
                                                          pow_p=pow_poly, funcs=ValsandNorms, derivative=derivative)
                else:
                    _, error_tensor[k][i, s] = LebesgueConst(x_tmp, col_exp * (ndim + 1), poly=poly_basis, test_pnts=points_test,
                                                             pow_p=pow_poly, funcs=ValsandNorms, derivative=derivative)
    return error_tensor


@jit
def maxvol_error(x, function, points_test, p_indices, n_col, mask, derivative=True):
    error_bmaxvol, leb_e = [], []
    ndim = points_test.shape[1]

    if type(function) is not list:
        function = [function]
    ValsandNorms = MakeValsAndNorms(function, points_test)
    for i in range(mask.shape[0]):
        rows_exp, col_exp = mask[i, 0], n_col[mask[i, 1]]
        p = p_indices[mask[i, 1]][:rows_exp]
        _, b = LebesgueConst(x[p], col_exp * (ndim + 1), poly=cheb, test_pnts=points_test, pow_p=1, funcs=ValsandNorms,
                             derivative=derivative)
        error_bmaxvol.append(b)
    return np.array(error_bmaxvol).reshape(-1)
