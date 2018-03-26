from __future__ import print_function
import numpy as np
from maxvolpy.maxvol import maxvol
from scipy.linalg import lu_factor
import scipy.linalg
import sys
from numba import jit


jit = lambda x : x

to_print_progress = False

def DebugPrint(s):
    if to_print_progress:
        print (s)
        sys.stdout.flush()

### how to make from permatation matrix readable array
_="""
def perm_array(A):
    # p_a = np.array((A.shape[0]))
    p_a = np.argmax(A, axis = 1)
    return p_a

def perm_matrix(p, m = 'P'):
    p_m = np.zeros((p.shape[0], p.shape[0]),  dtype=p.dtype) # Check type!!!!!
    if m == 'P':
        for i in range(0,p.shape[0]):     
            p_m[i,p[i]] = 1.0
    if m == 'Q':
        for i in range(0,p.shape[0]):     
            p_m[p[i],i] = 1.0
        
    return p_m

"""

@jit
def real_permute(p, sz=None):
    if sz is None:
        sz = max(len(p), np.max(p))

    idx = np.arange(sz)
    for i, e in enumerate(p):
        idx[i], idx[e] = idx[e], idx[i]

    # DebugPrint( "real_permute : {}<-{}, ({})".format(str(p), str(idx), sz) )

    return idx

_="""
def form_permute(C, j, ind):
    C[ind],C[j]=C[j],C[ind]
    return()  
"""

def p_preproc(p, ndim, overwrite_a=False):
    loc = p if overwrite_a else np.copy(p)
    for j in range(0, len(loc), ndim):
        loc[j:j+ndim].sort()
    return loc

@jit
def elimination(L,U,ind):
    k = L.shape[0]
    for i in range(ind+1, k):
        L[i,ind] = U[i,ind]/U[ind,ind]
        U[i,ind:] -= L[i,ind]*U[ind,ind:] 
    return ()

def plu(A):
    n, m = A.shape
    P = np.eye((n), dtype=float)
    L = np.eye(n, m, dtype=A.dtype)
    U = np.copy(A)
    for j in range(0, m):
        loc_max = np.argmax(np.abs(U[j:, j]))

        U[[j+loc_max,j],j:] = U[[j,j+loc_max], j:]
        
        L[[j+loc_max,j],:j] = L[[j,j+loc_max], :j]
        
        P[[j+loc_max,j],:] = P[[j,j+loc_max], :]
        
        
        for i in range(j+1, n):
            L[i,j] = U[i,j]/U[j,j]
            U[i,j:] -= L[i,j]*U[j,j:]
           
    return(P, L, U)

# Some opt. staff
def LU_opt(A):
    LU = A.copy()
    m = A.shape[0]
    p = np.arange(m)

    for k in xrange(m-1):
        i = np.argmax(np.abs(LU[p[k:], k])) + k
        p[[i, k]] = p[[k, i]]

        p_k = p[k]
        for p_j in p[k+1:]:
            LU[p_j, k   ]  = LU[p_j, k] / LU[p_k, k]
            LU[p_j, k+1:] -= LU[p_j, k] * LU[p_k, k+1:]

    return p, LU

@jit
def MakeLU_P(p, LU):
    n = p.size
    L = np.zeros_like(LU)
    U = np.zeros_like(LU)
    for i, p_i in enumerate(p):
        for j in xrange(n):
            if i == j:
                L[i, j] = 1
                U[i, j] = LU[p_i, j]
            else:
                if i < j:
                    U[i, j] = LU[p_i, j]
                else:
                    L[i, j] = LU[p_i, j]

    return np.eye(n, dtype=int)[p], L, U

def MakeLU(LU):
    L = np.tril(LU, -1)
    np.fill_diagonal(L, 1.0)
    return L, np.triu(LU[:LU.shape[1]])

# ---------------------------------------------------------

def pluq(A):
    n, m = A.shape
    p = np.arange(n)
    L = np.eye(n, m, dtype=A.dtype)
    U = np.copy(A)
    q = np.arange(m)
    yx = np.array([0, 0], dtype=int)
    
    for j in range(0, m):
        
        loc_max = np.argmax(np.abs(U[j:, j:]))
        yx[0] = loc_max / (m - j)
        yx[1] = loc_max % (m - j)
        
        
        ### Move Rows
        U[[j+yx[0],j],:] = U[[j,j+yx[0]], :]
        
        L[[j+yx[0],j],:j] = L[[j,j + yx[0]], :j]
        
        p[j+yx[0]],p[j] = p[j],p[j+yx[0]]
        
        ### Move Columns
        U[:,[j+yx[1],j]] = U[:,[j,j+yx[1]]]
        
        L[:j,[j+yx[1],j]] = L[:j,[j,j+yx[1]]]
        
        q[j+yx[1]],q[j] = q[j],q[j+yx[1]]
        
        
        for i in range(j+1, n):
            L[i,j] = U[i,j]/U[j,j]
            U[i,j:] -= L[i,j]*U[j,j:]    
            
    return(p, L, U[:m], q) 

def change_intersept(inew, iold):
    """
    change two sets of rows or columns when indices may intersept with preserving order
    RETURN two sets of indices,
    than say A[idx_n] = A[idx_o]
    """

    # DebugPrint(str(inew) + '<->' + str(iold))
    union = np.array(list( set(inew) | set(iold) ))
    idx_n = np.hstack((inew, np.setdiff1d(union, inew, assume_unique=True)))
    idx_o = np.hstack((iold, np.setdiff1d(union, iold, assume_unique=True)))
    return  idx_n.astype(int), idx_o.astype(int)

@jit
def det_search(A, ndim, start_ind, black_list):
        det = 0.0
        row = start_ind

        for k in range(start_ind, A.shape[0], ndim):
            if k not in black_list:
                #pair = A[k:k+ndim].T
                pair = np.rot90(A[k:k + ndim], 1, (1,0))
                ra = np.linalg.matrix_rank(pair,tol= 1e-13)
                _,s,_ = scipy.linalg.svd(pair)
                #print pair
                #print ra
                #print s

                if ra == ndim :
                    piv, _ = maxvol(pair)
                    cur_det = np.abs(np.linalg.det(pair[piv]))
                    if cur_det > det:
                        det, row = cur_det, k
                
        return(det, row)

def pluq_ids_old(A, nder = 1, debug = False):  
    
    def restore_lu(L,U,ind):
        k = L.shape[0]
        for i in range(ind+1, k):
            U[i,ind:] += L[i,ind]*U[ind,ind:]
        return (U)
    
    def restore_layer(L,U,ind, ndim):
        k = L.shape[0]
        for j in range(ndim + ind-1,ind-1, -1):
            for i in range(j+1, k):
                U[i,j:] += L[i,j]*U[j,j:]    
        return (U)
    
            
                
    n, m = A.shape
    P, Q = np.arange(n), np.arange(m)
    L = np.eye(n, m, dtype=A.dtype)
    U = np.copy(A)
    ndim = nder + 1
    black_list = []
    info = np.zeros((2), dtype=int)
    j = 0
    while (j < m):
        ### in each slice we are looking for 2x2 matrix with maxvol and memorize 'k' as a number of first row 
        ### 'k'- pair counter, which starts from current j position till the bottom of the matrix
        max_det, row_n = det_search(U,ndim,j,j,black_list)
        if (max_det == 0.0) and (j == 0):
            ### Critical error = no appropriate pair
            info[0] = 1
            return (P,L,U,Q,info)            
        if max_det == 0.0:
            if debug == True:
                print('error found')
            info[1] += 1
            j = j - ndim
            restore_layer(L,U,j,ndim)
            if debug:
            	print ('restored matrix')
            	print (U)
            max_det, row_n = det_search(U,ndim,j+ndim,j,black_list)
            if max_det == 0.0:
                # Critical error = all elements are in blacklist
                info[0] = 1
                return (P,L,U,Q,info)                
            black_list.append(row_n)
        loc_point = np.rot90(U[row_n:row_n + ndim][:,j:],1,(1,0))
        piv,_ = maxvol(loc_point)
        piv.sort()
         
        if (debug):
            print ('on the', j, 'slice')
            print ('best row block is', np.arange(ndim) + row_n)
            print ('column coordinates:', piv + j)
            print ('maxvol 2x2 submatrix', np.rot90(loc_point[piv],1,(0,1)))
            print ('with det = ', max_det)
            print ('pivoting and permutations start...')

        #print piv + j,np.arange(ndim) + j
        indx_n, indx_o = change_intersept(np.arange(ndim) + j,piv + j)
        U[:,indx_n] = U[:,indx_o]
        L[:j,indx_n] = L[:j,indx_o]
        Q[indx_n] = Q[indx_o]

        indx_n, indx_o = change_intersept(np.arange(ndim) + j,row_n + np.arange(ndim))
                
        U[indx_n,:] = U[indx_o,:]
        L[indx_n,:j] = L[indx_o,:j]
        P[indx_n] = P[indx_o]

        block = np.copy(U[j:j+ndim,j:j+ndim])
        #print block
        if (debug):
            if np.linalg.det(block) == max_det:
                print('correct 2x2 matrix')

        # p_loc, l, rt = lu(block)
        # p_loc  = perm_array(p_loc.T)

        _, p_loc = lu_factor(block, overwrite_a=True, check_finite=False)
        p_loc = real_permute(p_loc, ndim)


        #print p_loc,np.arange(ndim)+j,(np.arange(ndim)+j)[p_loc]
        indx_n, indx_o = change_intersept(np.arange(ndim)+j,(np.arange(ndim)+j)[p_loc])
        
        U[indx_n,:] = U[indx_o,:]
        L[indx_n,:j] = L[indx_o,:j]
        P[indx_n] = P[indx_o]
        #print ('after local perms')
        #print U[j:j+ndim]   

        

        if debug == True:
            print ('just before elim')
            print (U)
        if (debug):
            print('Elimination starts')
        ### make them all zeros! Below (j,j) element
        for idx in range(ndim):                      
            elimination(L,U,j + idx)
        if (debug):
            print('after {} eliminations'.format(ndim))
            print (U)

        j = j + ndim

      
    P_pr = p_preproc(P, ndim)
    return(P_pr,L,U,Q,info) 
#--------------------------------------------------------------

class SingularError(Exception):
    def __init__(self, value):
        self.value = value

#--------------------------------------------------------------
@jit
def det_search_index(A, Arow, Acol, ndim, start_ind1, start_ind2):
    det = 0.0
    row = start_ind1
    Acol_idx = Acol[start_ind2:]
    for k in range(start_ind1, A.shape[0], ndim):
          
        pair = np.rot90(A[Arow[k:k + ndim]][:, Acol_idx], 1, (1,0))
        rank = np.linalg.matrix_rank(pair)

        if rank == ndim:
            loc_piv, _ = maxvol(pair)
            cur_det = np.abs(np.linalg.det(pair[loc_piv]))
            if cur_det > det:
                det, row, final_piv = cur_det, k, loc_piv
                
    return(det, row)

@jit
def pluq_ids(A, nder, do_pullback=False, pullbacks = 10, debug = False, overwrite_a=False, preserve_order=True, small_matrix=True):
    n, m = A.shape
    P, Q = np.arange(n), np.arange(m)
    LU = A if overwrite_a else np.copy(A)
    ndim = nder + 1
    info = np.zeros(2, dtype=int)
    black_list = []
    extra_fl = False
    yx = np.zeros((2),dtype=int)
    if small_matrix:
        saveLU = np.empty((m//ndim, n, m))
        P_list = []
        Q_list = []
        def restore_layer(ind):
            # Add code here!!!
            pass
    else:
        def restore_layer(ind):
            # Add code here, but later
            pass

    j = 0
    while j < m:
        ### in each slice we are looking for 2x2 matrix with maxvol and memorize 'k' as a number of first row 
        ### 'k'- pair counter, which starts from current j position till the bottom of the matrix
        j_ndim =  j + ndim
        range_j_dim = np.arange(j, j_ndim)
        if do_pullback:
            if extra_fl==False :
                max_det, row_n = det_search(LU[:,j:], ndim, j, [])
            else:
                max_det = 0.0
            #print ('maxdet on pos stage ', max_det)
            if max_det == 0.0:

                while max_det == 0.0 :
                    if j == 0:
                        #critical error: no appropriate pair on the first step
                        info[1] = 1
                        #print ('before out ', P)
                        raise SingularError(info[1])
                    else:    
                        info[0] += 1
                        if info[0]>=pullbacks:
                            raise SingularError(info[0])
                        P,Q = np.copy(P_list[current_layer]),np.copy(Q_list[current_layer])
                        
                        LU = np.copy(saveLU[current_layer,:,:])
                        j -= ndim
                        j_ndim -= ndim
                        range_j_dim = np.arange(j, j_ndim)
                        max_det, row_n = det_search(LU[:,j:], ndim, j, black_list[current_layer])
                        #print ('max_det on pullback stage ', max_det)
                        if max_det == 0.0 :
                            current_layer -= 1
                            #print (black_list)
                            black_list.pop()
                            #print (black_list)
                            P_list.pop()
                            Q_list.pop()
                        else:
                            black_list[current_layer].append(row_n)   
                            #print (black_list)
                            extra_fl = False

            else:
                current_layer = j//ndim
                black_list.append([])
                black_list[current_layer].append(row_n)
                #print (black_list)                
                P_list.append(np.copy(P))
            
                Q_list.append(np.copy(Q))
                saveLU[current_layer,:,:] = np.copy(LU)
        else:        
            max_det, row_n = det_search(LU[:,j:], ndim, j, black_list)
            #print (row_n)
            if max_det == 0.0:
                if j == 0:
                    ### Critical error = no appropriate pair
                    info[0] = 1
                else:
                    info[0] = 2
                raise SingularError(info[0])

        # loc_point = LU[row_n:row_n + ndim][:, j:].T
        loc_point = np.rot90(LU[row_n:row_n + ndim][:, j:], 1, (1,0))
        piv, _ = maxvol(loc_point)
        piv.sort()

        ### Interchanging columns due to place ones forming maxvol submatrix into the upper left position
        indx_n, indx_o = change_intersept(range_j_dim, piv + j)
        LU[:, indx_n] = LU[:, indx_o]
        Q[indx_n] = Q[indx_o]

        ### Interchanging rows
        if j != row_n:
            indx_n, indx_o = change_intersept(range_j_dim, np.arange(row_n, row_n + ndim ))
            LU[indx_n] = LU[indx_o]
            P[indx_n] = P[indx_o]

        ### To avoid zeros on the main diagonal during the elimination process, we do local plu decomposition in the block
        #_, p_loc = lu_factor(LU[j:j_ndim][:,j:j_ndim], overwrite_a=False, check_finite=False)
        #p_loc = real_permute(p_loc, ndim)
        p_loc,_,_,q_loc = pluq(LU[j:j_ndim][:,j:j_ndim])
        ### Interchanging rows inside one block according to the local plu
        indx_n, indx_o = change_intersept(range_j_dim, p_loc + j)
        LU[indx_n] = LU[indx_o]
        P[indx_n] = P[indx_o]

        
        indx_n, indx_o = change_intersept(range_j_dim, q_loc + j)
        LU[:,indx_n] = LU[:,indx_o]
        Q[indx_n] = Q[indx_o]
        ### make them all zeros! Below (j,j) element
        for ind in range_j_dim:
                alpha = 1.0/LU[ind, ind] # Less accurate but faster
                for i in range(ind+1, n):
                    LU[i, ind]    *= alpha
                    LU[i, ind+1:] -= LU[i, ind]*LU[ind, ind+1:]

        j = j_ndim
        if j != m:
            loc_max = np.argmax(np.abs(LU[j:, j:]))
            yx[0] = loc_max / (m - j)
            yx[1] = loc_max % (m - j)
            elem_max =  LU[j+yx[0],j+yx[1]]
            #print elem_max
            #print LU[j-1,j-1]
            if np.abs(elem_max/LU[j-1,j-1])>5e+01 :
                extra_fl = True
    if preserve_order:
        p_preproc(P, ndim, overwrite_a=True)
    return P, Q, LU, info

if __name__ == '__main__':
    try:
        n = int(sys.argv[1])
    except:
        n = 5

    try:
        m = int(sys.argv[2])
    except:
        m = 3

    try:
        dim = int(sys.argv[3])
    except:
        dim = 3

    m += n
    n *= dim
    m *= dim

    while True:
        np.random.seed(42)
        A = np.random.randn(m, n)
        if np.linalg.matrix_rank(A) == n:
            break

    # A = np.eye(*A.shape)
    P, Q, LU, info = pluq_ids(A, dim-1, overwrite_a=False, preserve_order=False)
    L, U = MakeLU(LU)
    print ("error =", np.max(np.abs( L.dot(U) - A[P][:, Q]  )))
    print (P, Q)
    # print A
    # print L
    # print U


