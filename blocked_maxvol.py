import numpy as np

def SWM(C, deriv, ind_x, ind_j):
    B = np.copy(C)
    tmp_columns = np.copy(B[:,ind_j:ind_j + deriv + 1])
    tmp_columns[ind_j:ind_j + deriv + 1] -= np.eye(deriv + 1)
    tmp_columns[ind_x:ind_x + deriv + 1] -= np.eye(deriv + 1)
    
    b = B[ind_x:ind_x + deriv + 1][:,ind_j:ind_j + deriv + 1]
    
    tmp_rows = np.copy(B[ind_x:ind_x + deriv + 1])
    tmp_rows[:,ind_j:ind_j + deriv + 1] -= np.eye(deriv + 1)
    
    B -= np.dot(tmp_columns, np.dot(np.linalg.inv(b),tmp_rows))
    return B

def form_permute(C, j, ind):
    C[ind],C[j]=C[j],C[ind]
    return(C)  

def mov_row(C, j, ind_x):
    temp = np.copy(C[ind_x,:])
    C[ind_x,:] = C[j, :]
    C[j, :] = temp            
            

def block_maxvol(A_init, tol=0.05, max_iters=100, swm_upd = True):
# work on parameters
    ids_init = A_init[:A_init.shape[1]]
    temp_init = np.dot(A_init,np.linalg.inv(ids_init))

    A = np.copy(A_init)
    ids = np.copy(ids_init)
    temp = np.copy(temp_init)
    n = temp.shape[0]
    m = temp.shape[1]
    curr_det = np.abs(np.linalg.det(ids))
    Fl = True
    P = np.arange(n)
    index = np.zeros((2), dtype = int)
    iters = 0

    while Fl and (iters < max_iters) :
        max_det = 0.99
        for k in range(m,n,2):
                pair = temp[k:k+2]
                for j in range(0,m,2):
                    curr_det = np.abs(np.linalg.det(pair[:,j:j+2]))
                    if curr_det > max_det :
                        max_det = curr_det
                        index[0] = k
                        index[1] = j

        if (max_det) > (1 + tol):
            #Forming new permutation array
            form_permute(P,index[1],index[0])
            form_permute(P, index[1]+1, index[0]+1)

            ### Recalculating with new rows position
            if swm_upd == True:
                bljad = np.copy(temp)
                temp = SWM(bljad,1,index[0],index[1])
            else:    
                mov_row(A,index[1],index[0])
                mov_row(A,index[1] + 1,index[0] + 1)
                ids = A[:A.shape[1]]      
                temp = np.dot(A,np.linalg.inv(ids))           

            iters += 1
        else:
            Fl = False 
    return(A, temp, P)            