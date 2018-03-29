from __future__ import print_function
import numpy as np
from numba import jit

@jit
def SWM(B, ndim, i, j):
    tmp_columns = np.copy(B[:,j:j + ndim])
    tmp_columns[j:j + ndim] -= np.eye(ndim)
    tmp_columns[i:i + ndim] += np.eye(ndim)
    
    b = B[i:i + ndim][:,j:j + ndim]
    
    tmp_rows = np.copy(B[i:i + ndim])
    tmp_rows[:,j:j + ndim] -= np.eye(ndim)
    
    B -= np.dot(tmp_columns, np.dot(np.linalg.inv(b),tmp_rows))
    return (B)

def form_permute(C, j, ind):
    C[ind],C[j]=C[j],C[ind]
    return(C)  

def mov_row(C, j, ind_x):
    C[[ind_x,j],:] = C[[j,ind_x],:]

@jit
def block_maxvol(A_init, nder, tol=0.05, max_iters=100, swm_upd = True, debug = False):
# work on parameters
    n,m = A_init.shape
    ndim = nder + 1 
    if swm_upd:
        A = A_init
        ids = A_init[:m]
        B = np.dot(A_init,np.linalg.inv(ids))
    else:
        A = np.copy(A_init)
        ids = A[:m]
        B = np.dot(A,np.linalg.inv(ids))
                                    
    curr_det = np.abs(np.linalg.det(ids))
    Fl = True
    P = np.arange(n)
    index = np.zeros((2), dtype = int)
    iters = 0

    while Fl and (iters < max_iters) :
        max_det = 1.0
        for k in range(m,n,ndim):
                pair = B[k:k+ndim]
                for j in range(0,m,ndim):
                    curr_det = np.abs(np.linalg.det(pair[:,j:j+ndim]))
                    if curr_det > max_det :
                        max_det = curr_det
                        index[0] = k
                        index[1] = j

        if (max_det) > (1 + tol):
            #Forming new permutation array
            for idx in range(ndim):                          
                form_permute(P,index[1] + idx,index[0] + idx)
            
            if debug == True:
                print (P[:m])
            if (swm_upd == True) and (debug == True): 
                print('on the {} iteration with swm, pair {} {} chosen and pair{}'.format(iters,index[0],index[1],B[index[0]:index[0]+ndim][:,index[1]:index[1]+ndim]))
            if (swm_upd == False) and (debug == True):
                print('on the {} iteration with stan.oper, pair {} {} chosen and pair{}'.format(iters,index[0],index[1],B[index[0]:index[0]+ndim][:,index[1]:index[1]+ndim]))
            ### Recalculating with new rows position
            if swm_upd == True:
                B = SWM(B,ndim,index[0],index[1])
                #for idx in range(ndim):                      
                    #mov_row(A,index[1] + idx,index[0] + idx)
                
            else:    
                for idx in range(ndim):                      
                    mov_row(A,index[1] + idx,index[0] + idx)     
                B = np.dot(A,np.linalg.inv(ids))           

            iters += 1
        else:
            Fl = False 
    return (P)   
