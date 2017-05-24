import numpy as np
from maxvolpy.maxvol import maxvol

def perm_matrix(p, m = 'P'):
    p_m = np.zeros((p.shape[0],p.shape[0]),dtype=float)
    if m == 'P':
        for i in range(0,p.shape[0]):     
            p_m[i,p[i]] = 1.0
    if m == 'Q':
        for i in range(0,p.shape[0]):     
            p_m[p[i],i] = 1.0
        
    return p_m

def pluq_ids(A, debug = False):
    def mov_permute(C, j, ind):
        C[ind],C[j]=C[j],C[ind]
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
    
    def det_search(A,start_ind1, start_ind2):
        det = 0.0
        row = start_ind1 
        for k in range(start_ind1,A.shape[0],2):
            if k not in black_list:
                
                # pair = np.concatenate((A[k,start_ind2:],A[k+1,start_ind2:])).reshape(2,m-j).T
                pair = A[k:k+2][:,start_ind2:].T
                if np.linalg.matrix_rank(pair) == 2 :
                    piv,_ = maxvol(pair)
                    if np.abs(np.linalg.det(pair[piv])) > det:
                        det, row = np.abs(np.linalg.det(pair[piv])), k
        return(det, row)        
                
    n, m = A.shape[0], A.shape[1]
    P = np.arange(n)
    L = np.eye(n, m, dtype=float)
    U = np.copy(A)
    Q = np.arange(m)
    yx = np.array([0, 0], dtype=int)
    black_list = []
    info = np.zeros((2), dtype=int)
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
            if debug:
                print('error found')
            info[1] += 1
            j = j - 2
            restore_lu(L,U,j+1)
            restore_lu(L,U,j)
            print ('restored matrix')
            print (U)
            max_det, row_n = det_search(U,j+2,j)
            if max_det == 0.0:
                # Critical error = all elements are in blacklist
                info[0] = 1
                return (P,L,U,Q,info)
                
            black_list.append(row_n)
        pair = np.concatenate((U[row_n,j:],U[row_n + 1,j:])).reshape(2,m-j).T
        piv,_ = maxvol(pair)
        piv.sort() 
        
        diag = False
        if (pair[piv][0,0]== 0) or (pair[piv][1,1] == 0):
            yx[0] = row_n 
            yx[1] = piv[1]+ j
            diag = True
        else:
        
            yx[0] = row_n
            yx[1] = piv[0]+ j 
            
        if (debug):
            if np.linalg.det(pair[piv]) == max_det:
                print('correct 2x2 matrix')
              
            print ('on the', j, 'slice')
            print ('best row block is', row_n, row_n + 1)
            print ('column coordinates:', piv[0] + j, piv[1] + j)
            print ('maxvol 2x2 submatrix', pair[piv])
            print ('with det = ', max_det)
            print ('pivoting and permutations start...')
            
            
        ### U moving ###
        mov_LU(U,j,yx[0],yx[1])
        ####

        ### L moving ###
        mov_LU(L,j,yx[0],yx[1],m='L')
        ###
        
        ### P&Q moving ###
        mov_permute(P,j,yx[0])
        mov_permute(Q,j,yx[1])
        
     
        if (debug):
            print ('after 1st pivot')
            print (U)
                
        #choosing second element to pivot. 
        ### if true, it means we do not have to permute COLUMNS (but still have to permute rows) another one time, because it will return to the initial condition 
        if (j,j+1) == (piv[0]+j,piv[1]+j):
            yx[0] = row_n + 1
            ### U moving ###
            mov_LU(U,j+1,yx[0],j+1)
            ####
            if (debug):
                print ('after 2nd pivot, ')
                print U
            ### L moving ###
            mov_LU(L,j+1,yx[0],j+1,m='L')
            ###

            ### P&Q moving ###
            mov_permute(P,j+1,yx[0])  

        else:
            if (diag == True) and (j != piv[0]):                    
                yx[0] = row_n + 1
                yx[1] = piv[0]+ j 
            else:
                yx[0] = row_n + 1
                yx[1] = piv[1]+ j                 

            ### U moving ###
            mov_LU(U,j+1,yx[0],yx[1])
            ####
            if (debug):
                print ('after 2nd pivot')
                print U
            ### L moving ###
            mov_LU(L,j+1,yx[0],yx[1],m='L')
            ###

            ### P&Q moving ###
            mov_permute(P,j+1,yx[0])
            mov_permute(Q,j+1,yx[1])
        
        if (debug):
            print('Elimination starts')
        ### make them all zeros! Below (j,j) element
        elimination(L,U,j)
        if (debug):
            print('after 1st elimination')
            print U
        elimination(L,U,j+1)
        j = j + 2
        if (debug):
            print('after 2nd ilimination')
            print U
        
    return(P,L,U,Q,info)  
