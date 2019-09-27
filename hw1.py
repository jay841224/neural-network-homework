import numpy as np
import random
def hardlim(array):
    array[array>=0] = 1
    array[array<0] = 0
    return array
dict = {(0, 0): 1, (0, 1): 7, (1, 0): 6,(1, 1): 0}

P = np.array([[1, 0, 1, 1, 1, 1, 0, 0, 1],\
             [0, 0, 0, 1, 0, 0, 1, 1, 1],\
             [1, 1, 1, 0, 1, 1, 0, 1, 1],\
             [1, 1, 1, 1, 0, 1, 1, 1, 1]] )
T = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])


W = np.random.random((2, 9))*2-1
B = np.random.random((2, 1))
satisfy = False
i = 0
error = 0
#learn
while not satisfy:
    #transform P for n1
    p_tran_temp = P[i]
    p_tran = p_tran_temp.reshape(p_tran_temp.shape[0], 1)
    
    n1 = np.dot(W, p_tran)
    A = hardlim(n1)
    #transform T for E
    t_tran_temp = T[i]
    t_tran = t_tran_temp.reshape(t_tran_temp.shape[0], 1)

    E = t_tran - A
    #transform P for new W
    p_for_final = p_tran_temp.reshape(1, p_tran_temp.shape[0])
    
    W = W + np.dot(E, p_for_final)
    # calculate error
    error += E*E
    i += 1
    if i>3:
        if(error[0, 0]==0 and error[1, 0]==0):
            satisfy = True
        i = 0
        error = 0
    

print(W)
print(B)
#test
while(i<4):
    p_tran_temp = P[i]
    p_tran = p_tran_temp.reshape(p_tran_temp.shape[0], 1)
    
    n1 = np.dot(W, p_tran)
    A = hardlim(n1)
    ans = (A[0, 0], A[1, 0])
    print(dict[ans])
    i += 1




