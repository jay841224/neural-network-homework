import numpy as np
import random
P = np.array([[1, 1, 1, 1, -1, 1, 1, 1, 1],\
             [1, -1, 1, 1, 1, 1, -1, -1, 1],\
             [1, 1, 1, -1, 1, 1, -1, -1, 1],\
             [-1, -1, -1, 1, -1, -1, 1, 1, 1]])


T = np.array([[1, 1, 1, 1, -1, 1, 1, 1, 1],\
             [1, -1, 1, 1, 1, 1, -1, -1, 1],\
             [1, 1, 1, -1, 1, 1, -1, -1, 1],\
             [-1, -1, -1, 1, -1, -1, 1, 1, 1]])
ans_dic = {(1, 1, 1, 1, -1, 1, 1, 1, 1): 0, (1, -1, 1, 1, 1, 1, -1, -1, 1): 1, (1, 1, 1, -1, 1, 1, -1, -1, 1): 6, (-1, -1, -1, 1, -1, -1, 1, 1, 1): 7}

def hardlim(array):
    array[array>=0] = 1
    array[array<0] = -1
    return array

#shape(x,) to shape(x, 1)
def matrix_transform1(array):
    p_tran = array.reshape(array.shape[0], 1)
    return p_tran

#shape(x,) to shape(1, x)
def matrix_transform2(array):
    tran = array.reshape(1, array.shape[0])
    return tran
def learn():
    satisfy = False
    error_zero = True
    W = np.random.random((9, 9))*2 - 1
    i = 0
    error = 0
    #for x in range(4):
     #   W += np.dot(matrix_transform1(P[x]), matrix_transform2(P[x]))


    while not satisfy:
        a = hardlim(np.dot(W, matrix_transform1(P[i])))
        E = matrix_transform1(T[i]) - a
        error += E*E
        
        W += 0.1*(E)*P[i]
        i += 1
        if i > 3:
            for zero in error:
                if not zero == 0:
                    error_zero = False
            if error_zero == True:
                satisfy = True
            error_zero = True
            i = 0
            error = 0
    return W

def test(W):
    p = matrix_transform1(P[3])
    a = hardlim(np.dot(W, p))
    ans = []
    for x in a:
        ans.append(int(x))
    ans = tuple(ans)
    return(ans_dic[ans])
        

#main
print('the weight is:')
#learn weight
W = learn()
print(W)

print('this is test:')
#test weight
print(test(W))
