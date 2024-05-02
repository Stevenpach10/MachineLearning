
import numpy as np

def regression(x, W):
        res = [0] * len(x)
        for i in range (0, len(x)):
            sum = 0
            sum += W[0][0]
            for j in range(1, (len(W[i]))):
                sum += W[i][j] * x[i][0]**j
            res[i] = sum
        res = np.array(res)
        return res

def regression_Dev(x, W):
    res = [0] * len(x)
    for i in range (0, len(x)):
        sum = 0
        for j in range(1, (len(W[i]))):
            sum += W[i][j] * ((j) * x[i][0]**(j-1))
        res[i] = sum
    res = np.array(res)
    return res

x = np.array([
    [2]
])

W = np.array([
    [3,5,7,8]
])

print(regression(x, W))
print(regression_Dev(x, W))