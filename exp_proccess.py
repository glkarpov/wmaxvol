import numpy as np
import numpy.linalg as la
from maxvolpy.maxvol import rect_maxvol
from gen_mat import *
from gen_points import *
import re
import sys, getopt
import os
from mva_test import *
import pandas as pd
import random
import itertools
from ipywidgets import interactive, interact, widgets
# from sympy import *
from numba import jit


class experiment():
    def __init__(self, path_dspace, path_calculated_design, design_dim, out_dim):
        self.design_space_path = path_dspace
        self.ndim = design_dim
        self.out_dim = out_dim
        self.derivative = False
        extract_design = file_extraction((path_calculated_design))
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
        self.mask = index_preprocess(self.n_rows, self.expans, exp_mask, self.out_dim, step_along=1,
                                         new_extr_path=None)

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
        return (error)

    def set_weights_ones(self):
        for i, wght in enumerate(self.weights):
            card = len(wght)
            self.weights[i] = np.ones(card).tolist()

    def get_k_points(self, k, mode):
        if mode == "weights down":
            for i, p_ind in enumerate(self.p_indices):
                if len(p_ind) >= k:
                    Z = [x for _, x in sorted(zip(self.weights[i], p_ind))]
                    self.p_indices[i] = Z[:k]
                    self.weights[i] = sorted(self.weights[i])[:k]
        if mode == "equal_to_columns":
            for i, p_ind in enumerate(self.p_indices):
                Z = [x for _, x in sorted(zip(self.weights[i], p_ind))]
                self.p_indices[i] = Z[-self.expans[i]:]
                self.weights[i] = sorted(self.weights[i])[-self.expans[i]:]
                self.weight_iter = [sum(j) for i, j in enumerate(self.weights)]

    def change_expans(self, mode):
        if mode == "up_to_points":
            for i, p_ind in enumerate(self.p_indices):
                self.expans[i] = len(p_ind)

    def Lebesgue_mask(self, points_test):
        Leb_e = []
        taken_points = np.load(self.design_space_path + ".npz")
        x = taken_points['x']
        for i in range(self.mask.shape[0]):
            rows_exp, col_exp = self.mask[i, 0], self.expans[self.mask[i, 1]]
            p = self.p_indices[self.mask[i, 1]][:rows_exp]
            b = LebesgueConst(x[p], col_exp * (self.out_dim), poly=cheb, test_pnts=points_test, pow_p=1, wts=None,
                              funcs=None, derivative=self.derivative)
            Leb_e.append(b)
        bmv = np.array(Leb_e).reshape((1, len(Leb_e)))
        return (bmv)

    def Lebesgue_all(self, points_test, poly=cheb, wwts=True):
        if wwts:
            arg = self.weights
        else:
            arg = None
        Leb_e = []
        taken_points = np.load(self.design_space_path + ".npz")
        x = taken_points['x']
        for i, col_exp in enumerate(self.expans):
            p = self.p_indices[i]
            b = LebesgueConst(x[p], col_exp * (self.out_dim), poly=poly, test_pnts=points_test, pow_p=1, wts=None,
                              funcs=None, derivative=self.derivative)
            Leb_e.append(b)
        bmv = np.array(Leb_e).reshape((1, len(Leb_e)))
        return (bmv)


def index_preprocess(N_row, N_col, exp_mask, block_size, step_along=1, new_extr_path=None):
    if new_extr_path:
        N_row, N_col, p_indices = file_extraction(new_extr_path)
    indx = []
    for exp_indx, expsn in enumerate(N_col):
        N_r = N_row[exp_indx] / int(block_size)
        if expsn in exp_mask[0]:
            a = np.arange(expsn + 1, N_r + 1)
            indx += [np.array((i, j)) for i, j in zip(a.astype(int), itertools.repeat(exp_indx))]

        if exp_mask[-1] and int(np.around(exp_mask[-1] * expsn)) <= N_r:
            scaler = int(np.around(exp_mask[-1] * expsn))
            indx += ([np.array((scaler, exp_indx))])

    for pts_indx, pts in enumerate(exp_mask[1]):
        a = np.where(N_col < int(pts / block_size))[0]

        indx += [np.array((i, j)) for i, j in zip(itertools.repeat(int(pts / block_size)), a)]
    indx = np.array(indx)
    return (np.unique(indx, axis=0))

def submask(mask, crt_mask):
    rt = [(mask == crt_mask[i,:]).all(axis=1).nonzero()[0][0] for i in range(crt_mask.shape[0])]
    return(rt)


@jit
def multi_error_tensor(N_iter, N_col, mask,function, points_test, error_set, ndim, shape=None, derivative = True):
    block_size = ndim + 1
    error_tensor = np.empty((len(error_set), mask.shape[0], N_iter))
    if type(function) is not list:
            function = [function]
    ValsandNorms = MakeValsAndNorms(function, points_test)
    for s in np.arange(N_iter):
        print('Iteration #{}'.format(s))
        for i in range(mask.shape[0]):
            rows_exp, col_exp = mask[i,0], N_col[mask[i,1]]
            for k, points_type in enumerate(error_set):
                np.random.seed(s)
                x_tmp = complex_area_pnts_gen(rows_exp, ndim, mod=shape, distrib=points_type)
                if points_type=='sobol' and len((np.where(x_tmp == np.array([[0.,0.],])))[0])!=0:

                    while True:
                        x_tmp = complex_area_pnts_gen(rows_exp, ndim, mod=shape, distrib='sobol')
                        if len((np.where(x_tmp == np.array([[0.,0.],])))[0])==0:
                            break
                _, error_tensor[k][i,s] = LebesgueConst(x_tmp, col_exp*block_size, poly=cheb, test_pnts=points_test, pow_p=1, funcs=ValsandNorms, derivative = derivative)
    return(error_tensor)


@jit
def maxvol_error(x,function,points_test,p_indices,N_col, mask, derivative = True):
    error_bmaxvol, Leb_e = [], []
    ndim = points_test.shape[1]

    if type(function) is not list:
            function = [function]
    ValsandNorms = MakeValsAndNorms(function, points_test)
    print(mask)
    for i in range(mask.shape[0]):
        rows_exp, col_exp = mask[i,0], N_col[mask[i,1]]
        p = p_indices[mask[i,1]][:rows_exp]
        print(p)
        _,b = LebesgueConst(x[p], col_exp*(ndim+1), poly=cheb, test_pnts=points_test, pow_p=1, funcs=ValsandNorms, derivative = derivative)
        error_bmaxvol.append(b)
    bmv = np.array(error_bmaxvol).reshape((1,len(error_bmaxvol)))
    return(bmv)