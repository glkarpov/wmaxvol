import numpy as np
import numpy.linalg as la
import ids
from block_maxvol import *
# stuff to handle with matrix linings. Puts matrix U in lining, i.e. : B = A*UA or B = AUA*.
class lining:
    def __init__(self, A, U,inv = False):
        if inv == False:
            self.left = A.conjugate().T
            self.right = A
            self.core = U
        else:
            self.left = A
            self.right = A.conjugate().T
            self.core = U
    def left_prod(self):
        return np.dot(self.left, self.core)
    def right_prod(self):
        return np.dot(self.core, self.right)
    def assemble(self):
        return np.dot(self.left, self.right_prod())

# main func to form new coeff matrix    
def rect_core(C, C_sigma, ndim):
    inv_block = la.inv(np.eye(ndim) + np.dot(C_sigma, C_sigma.conjugate().T))
    puzzle = lining(C_sigma,inv_block)
    U = np.hstack((np.eye(C.shape[1]) - puzzle.assemble(), puzzle.left_prod()))
    C_new = lining(C,U,inv = True).left_prod()
    return C_new, puzzle.assemble()    

def form_permute(C, j, ind):
    C[ind],C[j]=C[j],C[ind]
    
def mov_row(C, j, ind_x):
    C[[ind_x,j],:] = C[[j,ind_x],:]
    
def cold_start(C, ndim):
    k = C.shape[0] // ndim
    n = C.shape[0]
    values = []
    for i in range(0,k):
        CC_T = np.dot(C[i*ndim:i*ndim+ndim], C[i*ndim:i*ndim+ndim].conjugate().T)
        values.append((CC_T))
    return values      
    
def rect_block_maxvol_core(A_init, nder, Kmax, t = 0.05):
    ndim = nder + 1
    M,n = A_init.shape[0], A_init.shape[1]
    block_n = M // ndim # Whole amount of blocks in matrix A
    P = np.arange(M) # Permutation vector
    Fl = True
    Fl_cs = True
    ids_init = A_init[:n]
    temp_init = np.dot(A_init,np.linalg.pinv(ids_init))

    A = np.copy(A_init)
    A_hat = np.copy(ids_init)
    C = np.copy(temp_init)
    
    shape_index = n
    C_w = np.copy(C)
    CC_sigma = []
    
    while (Fl == True) and (shape_index < Kmax):
            
        if (Fl_cs == False):
            ind_array = la.det(np.eye(ndim) + CC_sigma)
            elem = np.argmax(np.abs(ind_array[(shape_index // ndim):])) + (shape_index // ndim)
            print elem
            if (ind_array[elem] > 1 + t):
                CC_sigma[shape_index/ndim], CC_sigma[elem] = CC_sigma[elem], CC_sigma[shape_index/ndim]
                for idx in range(ndim):                          
                    form_permute(P,shape_index + idx,elem*ndim + idx)
                    mov_row(C_w,shape_index + idx,elem*ndim + idx)
                C_new, line = rect_core(C_w,C_w[shape_index:shape_index + ndim],ndim)
                #print C_new.shape, C_w.shape

                ### update list of CC_sigma
                for k in range(block_n):
                    CC_sigma[k] = CC_sigma[k] - lining(C_w[k*ndim:ndim*(k+1)],line,inv=True).assemble()
                C_w = C_new     
                shape_index += ndim
            else:
                print ('elements not found')
                Fl = False
                
        if Fl_cs:
            CC_sigma = cold_start(C_w, ndim)
            ind_array = la.det(np.eye(ndim) + CC_sigma)
            elem = np.argmax(np.abs(ind_array[(shape_index // ndim):])) + (shape_index // ndim)
            print elem
            if (ind_array[elem] > 1 + t):
                CC_sigma[shape_index/ndim], CC_sigma[elem] = CC_sigma[elem], CC_sigma[shape_index/ndim]
                for idx in range(ndim):                          
                    form_permute(P,shape_index + idx,elem*ndim + idx)
                    mov_row(C_w,shape_index + idx,elem*ndim + idx)
            else:
                print ('cold_start fail')
                Fl = False
            shape_index += ndim
            C_new, line = rect_core(C_w,C_w[n:shape_index],ndim)
            
            ### update list of CC_sigma
            for k in range(block_n):
                CC_sigma[k] = CC_sigma[k] - np.dot(C_w[k*ndim:ndim*(k+1)], np.dot(line, C_w[k*ndim:ndim*(k+1)].conjugate().T))
            C_w = C_new 
            Fl_cs = False      
            

    return(C_w, CC_sigma, P)     
    
def rect_block_maxvol(A, nder, Kmax, max_iters, rect_tol = 0.05, tol = 0.0, debug = False, ext_debug = False):
    pluq_perm,l,u,q,inf = ids.pluq_ids(A,nder, debug=False)
    A_init = np.dot(ids.perm_matrix(pluq_perm),np.dot(A,ids.perm_matrix(q)))
    A_rect_init,_,perm = block_maxvol(A_init, nder, tol = tol,max_iters=200,swm_upd=True)
    bm_perm = ids.perm_array(np.dot(ids.perm_matrix(perm),ids.perm_matrix(pluq_perm)))
    a, b, c = rect_block_maxvol_core(A_rect_init,nder,Kmax,t = rect_tol)
    final_perm = ids.perm_array(np.dot(ids.perm_matrix(c),ids.perm_matrix(bm_perm)))
    
    if ext_debug:
        return (a,b, final_perm, bm_perm, pluq_perm)
    else:
        return (final_perm)
        