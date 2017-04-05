import numpy as np
def plu(A):
    n, m = A.shape[0], A.shape[1]
    P = np.eye((n), dtype=float)
    L = np.eye((m), dtype=float)
    L_add = np.zeros((n-m, m), dtype=float)
    L = np.concatenate((L, L_add), axis = 0)
    U = np.copy(A)
    for j in range(0, m):
        loc_max = np.argmax(np.abs(U[j:, j]))
        temp = np.copy(U[j+loc_max,j:])
        U[j+loc_max,j:] = U[j, j:]
        U[j, j:] = temp

        temp = np.copy(L[j+loc_max,:j])
        L[j+loc_max,:j] = L[j, :j]
        L[j, :j] = temp    

        temp = np.copy(P[j+loc_max,:])
        P[j+loc_max,:] = P[j, :]
        P[j, :] = temp     
        
        for i in range(j+1, n):
            L[i,j] = U[i,j]/U[j,j]
            U[i,j:] -= L[i,j]*U[j,j:]
           
    return(P, L, U)       


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