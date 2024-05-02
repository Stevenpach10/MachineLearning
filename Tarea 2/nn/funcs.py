import numpy as np

def sigmoid(x):
    return 1.0/(1.0+np.exp(-x))

def sigmoid_grad(s):
    return s * (1.0 - s)


def relu(x):
    return x * (x > 0)

def  relu_grad(x):
    return 1.0 * (x > 0)

# Tanh
def tanh(x):
    return np.tanh(x)

# Tanh dev
#TANH (mx +b)
def tanh_grad(s):
    return 1 - np.tanh(s) ** 2

#with numerical stability
def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

def logloss(x, y):
    probs = softmax(x)
    return probs, -y * np.log(probs)

def logloss_grad(probs, y):
    probs[:,y] -= 1.0
    return probs

def batch_hits(x, y):
    return np.sum(np.argmax(x, axis=1) == y)

# Cross Enttropy Loss
def crossEntropyLoss(x, y):
    return -np.mean(np.sum(y * np.log(x + 1e-15), axis=1))

# MSE
def MSE(x, y):
    return 1/x.shape[0] * np.sum((x - y) ** 2)

# MSE dev
def MSE_grad(y, probs):
    res = 2/y.shape[0] * (y - probs)
    return res

# CrossEntropyLoss and Softmax derivates

def crossEntropySoftmax(x, y):
    return y - x