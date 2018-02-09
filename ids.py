import numpy as np
from maxvolpy.maxvol import maxvol
from scipy.linalg import lu as lu
import scipy.linalg 
from numba import jit

### how to make from permatation matrix readable array
def perm_array(A):
    p_a = np.array((A.shape[0]))
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

def form_permute(C, j, ind):
    C[ind],C[j]=C[j],C[ind]
    return()  

def p_preproc(p,ndim):
    loc = np.copy(p)
    for j in range(0,len(loc),ndim):
        loc[j:j+ndim] = np.sort(loc[j:j+ndim])
    return(loc)

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


def MakeLU(p, LU):
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

# ---------------------------------------------------------

def pluq(A):
    def mov_permute(C, j, ind, m = 'P'):
        if m == 'P':
            temp = np.copy(C[ind,:])
            C[ind,:] = C[j, :]
            C[j, :] = temp
        if m == 'Q':
            temp = np.copy(C[:,ind])
            C[:,ind] = C[:,j]
            C[:,j] = temp
        return()  
    def mov_LU(C, j, ind_r, ind_c, m = 'U'):
        if m == 'U':
            temp = np.copy(C[ind_r,:])
            C[ind_r,:] = C[j, :]
            C[j, :] = temp
        
            temp = np.copy(C[:,ind_c])
            C[:, ind_c] = C[:,j]
            C[:, j] = temp
        if m=='L':
            temp = np.copy(C[ind_r,:j])
            C[ind_r,:j] = C[j, :j]
            C[j, :j] = temp

            temp = np.copy(C[:j,ind_c])
            C[:j, ind_c] = C[:j,j]
            C[:j, j] = temp         
        return() 
    n, m = A.shape[0], A.shape[1]
    P = np.eye((n), dtype=float)
    L = np.eye((m), dtype=float)
    L_add = np.zeros((n-m, m), dtype=float)
    L = np.concatenate((L, L_add), axis = 0)
    U = np.copy(A)
    Q = np.eye((m), dtype=float)
    yx = np.array([0, 0], dtype=int)
    
    for j in range(0, m):
        
        loc_max = np.argmax(np.abs(U[j:, j:]))
        yx[0] = loc_max / (m - j)
        yx[1] = loc_max % (m - j)
        
        ### U moving ###
        mov_LU(U,j,j+yx[0],j+yx[1])
        ####
        
        ### L moving ###
        mov_LU(L,j,j+yx[0],j+yx[1],m='L')
        ###
        
        ### P&Q moving ###
        mov_permute(P,j,j+yx[0])
        mov_permute(Q,j,j+yx[1], m='Q')
        
        for i in range(j+1, n):
            L[i,j] = U[i,j]/U[j,j]
            U[i,j:] -= L[i,j]*U[j,j:]    
            
    return(P, L, U[:m], Q) 

def change_intersept(inew, iold, full=True):
    """
    change two sets of rows or columns when indices may intersept with preserving order
    RETURN two sets of indices,
    than say A[idx_n] = A[idx_o]
    """
    union = np.array(list( set(inew) | set(iold) ))
    idx_n = np.hstack((inew, np.setdiff1d(union, inew)))
    idx_o = np.hstack((iold, np.setdiff1d(union, iold)))
    return  idx_n.astype(int), idx_o.astype(int)

@jit
def det_search(A, ndim,start_ind1, start_ind2,black_list):
        det = 0.0
        row = start_ind1 

        for k in range(start_ind1,A.shape[0],ndim):
            if k not in black_list:
                #pair = A[k:k+ndim][:,start_ind2:].T
                pair = np.rot90(A[k:k + ndim][:,start_ind2:],1,(1,0))
                ra = np.linalg.matrix_rank(pair)

                if ra == ndim :
                    piv,_ = maxvol(pair)
                    if np.abs(np.linalg.det(pair[piv])) > det:
                        det, row = np.abs(np.linalg.det(pair[piv])), k
                
        return(det, row)

def pluq_ids(A, nder = 1, debug = False):  
    
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
        p_loc,l,rt = lu(block)

        p_loc  = perm_array(p_loc.T)

        #print p_loc,np.arange(ndim)+j,(np.arange(ndim)+j)[p_loc]
        indx_n, indx_o = change_intersept(np.arange(ndim)+j,(np.arange(ndim)+j)[p_loc])
        
        U[indx_n,:] = U[indx_o,:]
        L[indx_n,:j] = L[indx_o,:j]
        P[indx_n] = P[indx_o]
        #print ('after local perms')
        #print U[j:j+ndim]   

        

        if debug == True:
            print ('just before elim')
            print U
        if (debug):
            print('Elimination starts')
        ### make them all zeros! Below (j,j) element
        for idx in range(ndim):                      
            elimination(L,U,j + idx)
        if (debug):
            print('after {} eliminations'.format(ndim))
            print U

        j = j + ndim

      
    P_pr = p_preproc(P, ndim)
    return(P_pr,L,U,Q,info) 
#--------------------------------------------------------------
@jit
def det_search_index(A,Arow, Acol, ndim,start_ind1, start_ind2):
    det = 0.0
    row = start_ind1 
    final_piv = np.zeros(ndim)
    for k in range(start_ind1,A.shape[0],ndim):
          
        pair = np.rot90(A[Arow[k:k + ndim]][:,Acol[start_ind2:]],1,(1,0))
        rank = np.linalg.matrix_rank(pair)

        if rank == ndim :
            loc_piv,_ = maxvol(pair)
            if np.abs(np.linalg.det(pair[loc_piv])) > det:
                det, row = np.abs(np.linalg.det(pair[loc_piv])), k
                
    return(det, row)


def pluq_ids_index(A, nder, debug = False):  
   
    n, m = A.shape[0],A.shape[1]
    P, Q = np.arange(n), np.arange(m)
    L = np.eye(n, m, dtype=A.dtype)
    U = np.copy(A)
    Urow, Ucol  = np.arange(n), np.arange(m)
    Lrow, Lcol = np.arange(n), np.arange(m)
    ndim = nder + 1
  
    info = np.zeros((2), dtype=int)
    
    j = 0
    while (j < m):
        ### in each slice we are looking for 2x2 matrix with maxvol and memorize 'k' as a number of first row 
        ### 'k'- pair counter, which starts from current j position till the bottom of the matrix
        max_det, row_n = det_search_index(U,Urow,Ucol,ndim,j,j)
        if (max_det == 0.0) and (j == 0):
            ### Critical error = no appropriate pair
            info[0] = 1
            return (P,L,U,Q,info)            
        
        loc_point = np.rot90(U[Urow[row_n:row_n + ndim]][:,Ucol[j:]],1,(1,0))
        piv,_ = maxvol(loc_point)
        piv.sort()
        
        ### Interchanging columns due to place ones forming maxvol submatrix into the upper left position
        indx_n, indx_o = change_intersept(np.arange(ndim) + j,piv + j)
 
        Ucol[indx_n] = Ucol[indx_o]
        Lcol[indx_n] = Lcol[indx_o]           
        Q[indx_n] = Q[indx_o]

        ### Interchanging rows
        indx_n, indx_o = change_intersept(np.arange(ndim) + j,row_n + np.arange(ndim))
        Urow[indx_n] = Urow[indx_o]
        Lrow[indx_n] = Lrow[indx_o]           
        P[indx_n] = P[indx_o]
       
        ### To avoid zeros on the main diagonal during the elimination process, we do local plu decomposition in the block
        p_loc,l,rt = lu(U[Urow[j:j+ndim]][:,Ucol[j:j+ndim]])
        p_loc  = perm_array(p_loc.T)

        ### Interchanging rows inside one block according to the local plu
        indx_n, indx_o = change_intersept(np.arange(ndim)+j,(np.arange(ndim)+j)[p_loc])
        Urow[indx_n] = Urow[indx_o]
        Lrow[indx_n] = Lrow[indx_o]
        P[indx_n] = P[indx_o]
       
        ### make them all zeros! Below (j,j) element
        for idx in range(ndim):  
                ind = j + idx
                for i in range(ind+1, n):
                    L[Lrow[i]][Lcol[ind]] = U[Urow[i]][Ucol[ind]]/U[Urow[ind]][Ucol[ind]]
                    U[Urow[i]][Ucol[ind:]] -= L[Lrow[i]][Lcol[ind]]*U[Urow[ind]][Ucol[ind:]]
    
        if (debug):
            print('after {} eliminations'.format(ndim))
            print U

        j = j + ndim

      
    P_pr = p_preproc(P, ndim)
    return(P_pr.astype(int),L[Lrow][:,Lcol],U[Urow][:,Ucol],Q,info)  
