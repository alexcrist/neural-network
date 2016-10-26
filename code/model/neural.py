import numpy as np
import model.mathHelper as mh

class NeuralNetwork(object):
    ''' A neural network model with one hidden layer.

        Attributes: inputNodes  - number of nodes in input layer
                    hiddenNodes - number of nodes in hidden layer
                    outputNodes - number of nodes in output layer
                    penalty     - penalty value on network complexity (0 <= penalty < 1)
                    W1          - weight matrix for edges between input layer and hidden layer
                    W2          - weight matrix for edges between hidden layer and input layer
                    HPT         - "Hidden layer pre transform" matrix containing the values of nodes
                                  in the hidden layer before the sigmoid function is applied
                    H           - "Hidden layer" matrix containing the values of nodes in the hidden
                                  layer after the sigmoid function is applied
                    OPT         - "Output layer pre transform" matrix containing the values of nodes
                                  in the output layer before the sigmoid function is applied
                    O           - "Output layer" matrix containing the values of nodes in the output
                                  layer after the sigmoid function is applied '''

    # Creates a neural network
    def __init__(self, inputNodes, hiddenNodes, outputNodes, penalty):
        self.inputNodes = inputNodes
        self.hiddenNodes = hiddenNodes
        self.outputNodes = outputNodes
        self.penalty = penalty 

        # Randomly set weight matrices initially
        self.W1 = np.random.randn(self.inputNodes, self.hiddenNodes)
        self.W2 = np.random.randn(self.hiddenNodes, self.outputNodes)

    # Generate all nodes for a given set of input nodes
    def analyze(self, X):
        # Calculate the hidden nodes by taking the dot product of W1 and the input nodes
        self.HPT = np.dot(X, self.W1)
        # Apply the sigmoid function to the hidden nodes
        self.H = mh.sigmoid(self.HPT)

        # Calculate the output nodes by taking the dot product of W2 and the hidden nodes
        self.OPT = np.dot(self.H, self.W2)
        # Apply the sigmoid function to the output nodes
        self.O = mh.sigmoid(self.OPT)

    # Cost for a set of inputs and outputs
    def cost(self, X, Y):
        self.analyze(X)
        cost = 0.5 * sum((Y - self.O)**2) / X.shape[0] + 0.5 * self.penalty * (np.sum(self.W1**2) + np.sum(self.W2**2))
        return cost

    # Cost derivative for a set of inputs and outputs
    def costDeriv(self, X, Y):
        self.analyze(X)

        delta2 = np.multiply(-(Y - self.O), mh.sigmoidDeriv(self.OPT))
        delta1 = np.dot(delta2, self.W2.T) * mh.sigmoidDeriv(self.HPT)

        dCdW1 = np.dot(X.T, delta1)/X.shape[0] + self.penalty * self.W1
        dCdW2 = np.dot(self.H.T, delta2)/X.shape[0] + self.penalty * self.W2

        return dCdW1, dCdW2

    # Cost gradient for a set of inputs and outputs
    def computeGradient(self, X, Y):
        dCdW1, dCdW2 = self.costDeriv(X, Y)
        return np.concatenate((dCdW1.ravel(), dCdW2.ravel()))

    # Returns all weight matrices as one flat array
    def getWeights(self):
        return np.concatenate((self.W1.ravel(), self.W2.ravel()))

    # Sets the weight matrices from one flat array
    def setWeights(self, weightArray):
        W1_start = 0
        W1_end = self.inputNodes * self.hiddenNodes
        W2_start = W1_end
        W2_end = W2_start + self.hiddenNodes * self.outputNodes

        self.W1 = np.reshape(weightArray[W1_start:W1_end], (self.inputNodes, self.hiddenNodes))
        self.W2 = np.reshape(weightArray[W2_start:W2_end], (self.hiddenNodes, self.outputNodes))