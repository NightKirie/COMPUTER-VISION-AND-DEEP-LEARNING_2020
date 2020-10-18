import numpy as np
a = np.array([[1,2,3],[4,5,6],[7,8,9]])
b =  np.array([[0,0,0],[0,0,0],[0,0,0]])
for i in range(0, a.shape[0]):
    for j in range(0, a.shape[1]):
        if a[i, j] < a.max() * 0.3:
            b[i, j] = 1
print(b)
