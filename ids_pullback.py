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

def pluq_ids_pull(A, debug = False):
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
    
    def restore_layer(L,U,ind):
        k = L.shape[0]
        down_ind = ind + 1
        for i in range(down_ind+1, k):
            U[i,down_ind:] += L[i,down_ind]*U[down_ind,down_ind:]
        for i in range(ind+1, k):
            U[i,ind:] += L[i,ind]*U[ind,ind:]    
        return (U)
    
    def det_search(A,start_ind1, start_ind2):
        det = 0.0
        row = start_ind1 
        for k in range(start_ind1,A.shape[0],2):
            if k not in black_list[curr_layer]:
                pair = A[k:k+2][:,start_ind2:].T
                if np.linalg.matrix_rank(pair) == 2 :
                    piv,_ = maxvol(pair)
                    if np.abs(np.linalg.det(pair[piv])) > det:
                        det, row = np.abs(np.linalg.det(pair[piv])), k
        return(det, row)      
    
    def pullback(err_ind,L,U, curr_layer):
        if (err_ind < 0):
            return(None, None, None)        
        restore_layer(L,U,err_ind)
        max_det, row_n = det_search(U,err_ind + 2,err_ind)
        if max_det == 0.0:
            # Critical error = all elements are in blacklist or det-zeros
            info[0] = 1
            curr_layer -= 1
            black_list.pop()
            info[3] += 1
            err_ind = err_ind - 2 ### pullback itself. Change position of the slice pointer (j) 
            pullback(err_ind, L,U)
        ### But if we managed to find another good pair, we add its index to black_list (swap will be later, but we  will never take this entry again on this layer. And return parametres to the outer programm.    
        else:
            black_list[(curr_layer)].append(row_n)
            info[1] += 1
            return(max_det,row_n, err_ind,curr_layer)
    n, m = A.shape[0], A.shape[1]
    P = np.arange(n)
    L = np.eye(n, m, dtype=float)
    U = np.copy(A)
    Q = np.arange(m)
    yx = np.array([0, 0], dtype=int)
    black_list = [[]]
    info = np.zeros((4), dtype=int)
    treshold = 1e-10
    curr_layer = -1
    j = -2
    while (j < m - 2):
        ### in each slice we are looking for 2x2 matrix with maxvol and memorize 'k' as a number of first row 
        ### 'k'- pair counter, which starts from current j position till the bottom of the matrix
        
        max_det, row_n = det_search(U,j+2,j+2)           
        if max_det == 0.0:
            info[3] += 1
            if debug:
                print('error found')
                print('pullback starts on layer', curr_layer)
            max_det, row_n, etr,curr_layer = pullback(j, L, U,curr_layer)
            
            if (max_det,row_n) == (None, None) :
                return(None,None,None,None,info)
            j = etr
        else:    
            j = j + 2  
            curr_layer += 1
            
            if j > 0:   ### its just to avoid appending on the very first layer. We state that on the layer-0 always exists at least 1 good pair.
                black_list.append([])
                info[2] += 1
            if debug:
                print('Jumped on the layer', curr_layer)
                print('black_list now contains entries = ', len(black_list))
                  
        pair = U[row_n:row_n + 2][:,j:].T
        piv,_ = maxvol(pair)
        piv.sort() 
        ### Replacement part! Dont need to know about processes above
        diag = False
        if (np.abs(pair[piv][0,0]) < treshold) or (np.abs(pair[piv][1,1]) < treshold):
            yx[0] = row_n 
            yx[1] = piv[1]+ j
            diag = True
            if debug:
                print ('diag case')
                print (pair[piv])
        else:
        
            yx[0] = row_n
            yx[1] = piv[0]+ j
            
        if (debug):
            if np.linalg.det(pair[piv]) == max_det:
                print('correct 2x2 matrix')
              
            print ('on the', j, 'slice; curr_layer = ', curr_layer)
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
        if (0,1) == (piv[0],piv[1]):
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
            if (diag == True) and (piv[0] != 0):                    
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
        
        if (debug):
            print('after 2nd ilimination')
            print U
        
    return(P,L,U,Q,info)  
