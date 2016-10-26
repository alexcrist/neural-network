import numpy as np

# Sigmoid function. Used as a simple activation function in a neural network
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Derivative of the sigmoid function
def sigmoidDeriv(x):
    return np.exp(-x) / ((1 + np.exp(-x))**2)