import numpy as np
import numpy.linalg as LA
from scipy import optimize
from maxvolpy.maxvol import rect_maxvol, maxvol
import gen_mat as gen
import grad_log_det as log_det # this is a module with analytical calculation of gradient and objective 
from mva_test import test_points_gen
from numba import jit, njit, prange
from mva_test import *
import os
import itertools
#get_ipython().magic('matplotlib inline')


def MV(num_col=50, num_points=54, interval=(-1.0,1.0), poly=gen.cheb):
    big = test_points_gen(int(1e5), 2, interval=interval, distrib='LHS')
    M = gen.GenMat(num_col, big, poly = poly, ToGenDiff=False) 
    row_indx,_ = rect_maxvol(M, tol = 1.0095, minK = num_points, maxK = num_points)
    pnts = M[row_indx][:,1:3]
    return pnts

def GD(num_col=50, num_points=54, num_tries=3, interval=(-1.0,1.0), poly=gen.cheb):
    log_det.num_col = num_col
    log_det.dim = dim  
    bnds = (interval,) * (dim*num_points)
    dstrs = ['LHS']
        
    loss_f = lambda x: log_det.loss_func(x, poly=poly, ToGenDiff=False)
    loss_grad = lambda x: log_det.grad(x, poly=poly)
    res = np.inf
    for distrib in dstrs:
        for _ in range(num_tries):
            x_0 = test_points_gen(num_points, dim, interval=interval, distrib=distrib) # starting point of GD
            x_0 = x_0.ravel('F')
            op = optimize.fmin_l_bfgs_b(loss_f, x_0, fprime = loss_grad, factr = 10.0, bounds = bnds)
            res_cur = op[1]
            if res_cur < res:
                res = res_cur
                pnts = op[0]
            
    pnts = pnts.reshape(pnts.size//dim, dim, order="F")             
    return pnts

def Experiment(dim=2, col=(10, 20), n_factor=np.linspace(1.0,3.0,4), interval=(-1.0, 1.0),
               polys=[gen.cheb, gen.legendre],
               repetitions = 50,
               sampling = ['LHS','Sobol','MV','GD'],
               wdir='./oversampling_new'):
    # create a new working directory and write all the files into it
    try:
        os.mkdir(wdir)
    except:
        pass
    
    polin = ['cheb','legendre']
    
    log_det.dim = dim  # this is a global variable in module log_det
    
    #p = polys[0]
    for index,p in enumerate(polys):
        fpath = os.path.join(wdir, "sincos_col={}_col={}_poly={}.txt").format(col[0], col[1], polin[index])
        file = open(fpath,'ab')
        for s in sampling:
            INFTY_error = np.zeros((col[1]-col[0]+1, repetitions))
            for l in range(col[0],col[1]+1):
                print("l = ", l)
                for rep in prange(repetitions):
                    if s == 'LHS':
                        pnts = test_points_gen(int(np.ceil(l)),dim, interval=interval, distrib=s, criterion='m', iterations=20)
                        _,INFTY_error[l-col[0],rep] =                        LebesgueConst(pnts, l, poly=p, test_pnts=test, pow_p=1, funcs=ff, derivative = False)
                    if s == 'Sobol':
                        pnts = test_points_gen(int(np.ceil(l)),dim, interval=interval, distrib=s, seed=rep)
                        _,INFTY_error[l-col[0],rep] =                        LebesgueConst(pnts, l, poly=p, test_pnts=test, pow_p=1, funcs=ff, derivative = False)
                    if s == 'MV':
                        pnts = MV(l, int(np.ceil(l)), interval=interval, poly=p)
                        _,INFTY_error[l-col[0],rep] =                        LebesgueConst(pnts, l, poly=p, test_pnts=test, pow_p=1, funcs=ff, derivative = False)
                    if s == 'GD':
                        pnts = GD(l, int(np.ceil(l)), interval=interval, poly=p)
                        _,INFTY_error[l-col[0],rep] =                        LebesgueConst(pnts, l, poly=p, test_pnts=test, pow_p=1, funcs=ff, derivative = False)
            print(str(s),'\n',INFTY_error)
            np.savetxt(file,INFTY_error, fmt='%.7e',header='INFTY '+str(s),footer='\n')
            file.flush()  
        file.close()

np.random.seed(11)
dim = 2
interval = (-1.0, 1.0)
test = test_points_gen(int(1e6), dim, interval=interval, distrib='LHS')
ff = MakeValsAndNorms([f_sincos], test)
Experiment(col=(30, 48), interval=interval)

