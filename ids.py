import numpy as np
from maxvolpy.maxvol import maxvol
def pluq_ids(A, debug = True):
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
            
    def elimination(L,U,ind):
        k = L.shape[0]
        for i in range(ind+1, k):
            L[i,ind] = U[i,ind]/U[ind,ind]
            U[i,ind:] -= L[i,ind]*U[ind,ind:] 
        return ()    
    
    n, m = A.shape[0], A.shape[1]
    P = np.eye((n), dtype=float)
    L = np.eye((m), dtype=float)
    L_add = np.zeros((n-m, m), dtype=float)
    L = np.concatenate((L, L_add), axis = 0)
    U = np.copy(A)
    Q = np.eye((m), dtype=float)
    yx = np.array([0, 0], dtype=int)
    max_det = np.zeros((2), dtype=float)
    
    
    for j in range(0, m, 2):
        ### in each slice we are looking for 2x2 matrix with maxvol and memorize 'k' as a number of first row 
        ### 'k'- pair counter, which starts from current j position till the bottom of the matrix
        max_det = np.zeros((2), dtype=float)
        for k in range(j,n,2):
            pair = np.concatenate((U[k,j:],U[k+1,j:])).reshape(2,m-j).T
            piv,_ = maxvol(pair)
            print pair
            if np.abs(np.linalg.det(pair[piv])) > max_det[0]:
                max_det[0], max_det[1] = np.abs(np.linalg.det(pair[piv])), k
         
        pair = np.concatenate((U[max_det[1].astype(int),j:],U[max_det[1].astype(int)+1,j:])).reshape(2,m-j).T
        piv,_ = maxvol(pair)
        piv.sort() 
        
        diag = False
        if (pair[piv][0,0]== 0) or (pair[piv][1,1] == 0):
            yx[0] = max_det[1] 
            yx[1] = piv[1]+ j
            diag = True
        else:
        
            yx[0] = max_det[1] 
            yx[1] = piv[0]+ j 
            
        if (debug):
            if np.linalg.det(pair[piv]) == max_det[0]:
                print('correct 2x2 matrix')
              
            print ('on the', j, 'slice')
            print ('best row block is', max_det[1].astype(int), max_det[1].astype(int) + 1)
            print ('column coordinates:', piv[0] + j, piv[1] + j)
            print ('maxvol 2x2 submatrix', pair[piv])
            print ('with det = ', max_det[0])
            print ('pivoting and permutations start...')
            
            
        ### U moving ###
        mov_LU(U,j,yx[0],yx[1])
        ####
        print U
        ### L moving ###
        mov_LU(L,j,yx[0],yx[1],m='L')
        ###
        
        ### P&Q moving ###
        mov_permute(P,j,yx[0])
        mov_permute(Q,j,yx[1], m='Q')
        
        ### make them all zeros! Below (j,j) element
        elimination(L,U,j)     
        print U
        #choosing second element to pivot. 
        ### if true, it means we do not have to permute another one time, because it will return to the initial condition 
        if (j,j+1) == (piv[0]+j,piv[1]+j):
            
            elimination(L,U,j+1)
        else:
            if diag == True:
                yx[0] = max_det[1] + 1
                yx[1] = piv[0]+ j 
            else:
                yx[0] = max_det[1] + 1
                yx[1] = piv[1]+ j                 

            ### U moving ###
            mov_LU(U,j+1,yx[0],yx[1])
            ####
            print U
            ### L moving ###
            mov_LU(L,j+1,yx[0],yx[1],m='L')
            ###

            ### P&Q moving ###
            mov_permute(P,j+1,yx[0])
            mov_permute(Q,j+1,yx[1], m='Q')

            ### make them all zeros! (Below (j+1,j+1) element) ###
            elimination(L,U,j+1)
        print U
    return(P,L,U,Q)  