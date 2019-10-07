import numpy as np
import random
def hardlim(array):
    array[array>=0] = 1
    array[array<0] = 0
    return array

#shape(x,) to shape(x, 1)
def matrix_transform1(array):
    p_tran = array.reshape(array.shape[0], 1)
    return p_tran

#shape(x,) to shape(1, x)
def matrix_transform2(array):
    tran = array.reshape(1, array.shape[0])
    return tran

dict = {(0, 0): 1, (0, 1): 7, (1, 0): 6,(1, 1): 0}

P = np.array([[1, 0, 1, 1, 1, 1, 0, 0, 1],\
             [0, 0, 0, 1, 0, 0, 1, 1, 1],\
             [1, 1, 1, 0, 1, 1, 0, 1, 1],\
             [1, 1, 1, 1, 0, 1, 1, 1, 1]] )
T = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])

def learn(matrix):
    W = np.random.random((matrix[1], matrix[0]))*2 - 1
    B = np.random.random((matrix[0], 1))
    satisfy = False
    i = 0
    error = 0
    #learn
    while not satisfy:
        #transform P for n1
        p_tran = matrix_transform1(P[i])
        
        n1 = np.dot(W, p_tran)
        A = hardlim(n1)
        #transform T for E
        t_tran = matrix_transform1(T[i])

        E = t_tran - A
        #transform P for new W
        p_for_final = matrix_transform2(P[i])
        
        W = W + np.dot(E, p_for_final)
        # calculate error
        error += E*E
        i += 1
        if i > 3:
            if(error[0, 0] == 0 and error[1, 0]==0):
                satisfy = True
            i = 0
            error = 0
    return [W, B]
    

def test(tolearn):
    i = 0
    W = tolearn[0]
    while(i<4):
        p_tran_temp = P[i]
        p_tran = p_tran_temp.reshape(p_tran_temp.shape[0], 1)
        
        n1 = np.dot(W, p_tran)
        A = hardlim(n1)
        ans = (A[0, 0], A[1, 0])
        print(dict[ans])
        i += 1



tolearn = learn([9, 2])
print('weight is:')
print((tolearn[0]))
print('biases is:')
print((tolearn[1]))
#test
test(tolearn)


