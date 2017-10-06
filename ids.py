import numpy as np
from maxvolpy.maxvol import maxvol
from scipy.linalg import lu as lu
import scipy.linalg 

### how to make from permatation matrix readable array
def perm_array(A):
    p_a = np.array((A.shape[0]))
    p_a = np.argmax(A, axis = 1)
    return p_a

def perm_matrix(p, m = 'P'):
    p_m = np.zeros((p.shape[0],p.shape[0]),dtype=float)
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

def change_intersept(inew, iold, full=True):
    """
    change two sets of rows or columns when indices may intersept with preserving order
    RETURN two sets of indices,
    than say A[idx_n] = A[idx_o]
    """
    union = np.array(list( set(inew) | set(iold) ))
    idx_n = np.hstack((inew, np.setdiff1d(union, inew)))
    idx_o = np.hstack((iold, np.setdiff1d(union, iold)))
    return  idx_n, idx_o

def pluq_ids(A, nder = 1, debug = False):

    def mov_LU(C, j, ind_r, ind_c, m = 'U'):
        if m == 'U':
            C[[ind_r,j],:] = C[[j,ind_r],:]
            C[:,[ind_c,j]] = C[:,[j,ind_c]]
        if m=='L':
            temp = np.copy(C[ind_r,:j])
            C[ind_r,:j] = C[j, :j]
            C[j, :j] = temp

            temp = np.copy(C[:j,ind_c])
            C[:j, ind_c] = C[:j,j]
            C[:j, j] = temp  
            
    def elimination(L,U,ind):
        k = L.shape[0]
        for i in range(ind+1, k):
            L[i,ind] = U[i,ind]/U[ind,ind]
            U[i,ind:] -= L[i,ind]*U[ind,ind:] 
        return ()    
    
    def restore_lu(L,U,ind):
        k = L.shape[0]
        for i in range(ind+1, k):
            U[i,ind:] += L[i,ind]*U[ind,ind:]
        return (U)
    
    def restore_layer(L,U,ind, ndim):
        k = L.shape[0]
        for j in range(ndim-1,ind-1, -1):
            for i in range(j+1, k):
                U[i,ind:] += L[i,ind]*U[ind,ind:]    
        return (U)
    
    def det_search(A,start_ind1, start_ind2):
        det = 0.0
        row = start_ind1 

        for k in range(start_ind1,A.shape[0],ndim):
            if k not in black_list:
                pair = A[k:k+ndim][:,start_ind2:].T
                #print pair, np.linalg.matrix_rank(pair)
                _,y,_ = scipy.linalg.svd(pair)
                ra = 0
                #print y
                for t in range(0,len(y)):
                    if np.abs(y[t]) > 1e-20:
                        #print ra
                        ra = ra + 1
                if ra == ndim :
                    piv,_ = maxvol(pair)
                    if np.abs(np.linalg.det(pair[piv])) > det:
                        det, row = np.abs(np.linalg.det(pair[piv])), k
                
        return(det, row)        
                
    n, m = A.shape[0], A.shape[1]
    P = np.arange(n)
    L = np.eye(n, m, dtype=float)
    U = np.copy(A)
    Q = np.arange(m)
    ndim = nder + 1
    black_list = []
    info = np.zeros((2), dtype=int)
    treshold = 1e-10
    j = 0
    while (j < m):
        ### in each slice we are looking for 2x2 matrix with maxvol and memorize 'k' as a number of first row 
        ### 'k'- pair counter, which starts from current j position till the bottom of the matrix
        max_det, row_n = det_search(U,j,j)
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
            max_det, row_n = det_search(U,j+ndim,j)
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

        print piv + j,np.arange(ndim) + j
        indx_n, indx_o = change_intersept(np.arange(ndim) + j,piv + j)
        U[:,indx_n] = U[:,indx_o]
        L[:j,indx_n] = L[:j,indx_o]           
        Q[indx_n] = Q[indx_o]
                   
      
        #print Q
        indx_n, indx_o = change_intersept(np.arange(ndim) + j,row_n + np.arange(ndim))
                
        U[indx_n,:] = U[indx_o,:]
        L[indx_n,:j] = L[indx_o,:j]           
        P[indx_n] = P[indx_o]
        #print('rows swapped')
        #print P
        block = np.copy(U[j:j+ndim,j:j+ndim])
        print block
        if (debug):
            if np.linalg.det(block) == max_det:
                print('correct 2x2 matrix')
        p_loc,l,rt = lu(block)
        print p_loc.T
        #print la.det(rt)
        p_loc  = perm_array(p_loc.T)
        print ('p_locs')
        print p_loc,np.arange(ndim)+j,(np.arange(ndim)+j)[p_loc]
        indx_n, indx_o = change_intersept(np.arange(ndim)+j,(np.arange(ndim)+j)[p_loc])
        #print indx_n,indx_o
        #aab = np.concatenate((indx_n,indx_o))
        #bba = np.concatenate((indx_o,indx_n))
        #print aab,bba
        
        U[indx_n,:] = U[indx_o,:]
        L[indx_n,:j] = L[indx_o,:j]
        P[indx_n] = P[indx_o]
        print ('after local perms')
        print U[j:j+ndim]   
        #print P[j:j+ndim]
        
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
        #print np.linalg.matrix_rank(U)
      

    return(P,L,U,Q,info)  