import numpy as np
import matplotlib.pyplot as plt
import mathHelper as mh
import csv
from scipy import optimize

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

    # Calculate the cost for a set of inputs and outputs
    def cost(self, X, Y):
        self.analyze(X)
        cost = 0.5 * sum((Y - self.O)**2) / X.shape[0] 
             + 0.5 * self.penalty * (np.sum(self.W1**2) + np.sum(self.W2**2))
        return cost

#
    def costDeriv(self, X, Y):
        self.analyze(X)

        delta2 = np.multiply(-(Y - self.O), mh.sigmoidDeriv(self.OPT))
        delta1 = np.dot(delta2, self.W2.T) * mh.sigmoidDeriv(self.HPT)

        dCdW1 = np.dot(X.T, delta1)/X.shape[0] + self.penalty * self.W1
        dCdW2 = np.dot(self.H.T, delta2)/X.shape[0] + self.penalty * self.W2

        return dCdW1, dCdW2

    def computeGradient(self, X, Y):
        dCdW1, dCdW2 = self.costDeriv(X, Y)
        return np.concatenate((dCdW1.ravel(), dCdW2.ravel()))

    def getWeights(self):
        return np.concatenate((self.W1.ravel(), self.W2.ravel()))

    def setWeights(self, weightArray):
        W1_start = 0
        W1_end = self.inputNodes * self.hiddenNodes
        W2_start = W1_end
        W2_end = W2_start + self.hiddenNodes * self.outputNodes

        self.W1 = np.reshape(weightArray[W1_start:W1_end], (self.inputNodes, self.hiddenNodes))
        self.W2 = np.reshape(weightArray[W2_start:W2_end], (self.hiddenNodes, self.outputNodes))

class Trainer(object):
    ''' A trainer used to train a neural network to a dataset.

        Attributes: network - a neural network model '''

    def __init__(self, network):
        self.network = network

    def costWrapper(self, weightArray, X, Y):
        self.network.setWeights(weightArray)
        cost = self.network.cost(X, Y)
        gradient = self.network.computeGradient(X, Y)
        return cost, gradient

    def train(self, X, Y):
        self.X = X
        self.Y = Y
        self.costs = []

        weightArray = self.network.getWeights()

        result = optimize.minimize(self.costWrapper, 
            weightArray, 
            jac=True, 
            method='BFGS', 
            args=(X, Y), 
            options={'maxiter': 1000, 'disp': True})

        self.network.setWeights(result.x)
        self.optimizationResults = result

class NeuralNetworkHelper(object):
    ''' A helper class that builds a trained network.

        Attributes: network - a neural network
                    trainX  - training input data
                    trainY  - training output data
                    xMax    - max value of training input data
                    yMax    - max value of training output data '''

    def __init__(self, trainX, trainY, penalty=0.0001):
        m, n = trainX.shape
        o, p = trainY.shape

        self.network = NeuralNetwork(n, n + 1, p, penalty)
        self.xMax = np.amax(trainX, axis=0)
        self.yMax = np.amax(trainY, axis=0)

        self.trainX = trainX / self.xMax
        self.trainY = trainY / self.yMax

        trainer = Trainer(self.network)
        trainer.train(self.trainX, self.trainY)

    # Returns a set of generated output guesses for given input data
    def analyze(self, testX):
        # Transform input data to [0 - 1] scale
        testX = testX / self.xMax

        # Analyze data
        self.network.analyze(testX)

        # Untransform data back from [0 - 1] scale
        output = self.network.O * self.yMax

        return output

    def getPenaltyCost(self, testX, testY, penalty):
        costSum = 0
        trials = 10
        for i in range(trials):
            network = NeuralNetworkHelper(self.trainX, self.trainY, penalty=penalty)
            output = network.analyze(testX)
            costSum += np.sum(abs(output - testY))
        return costSum / trials

    def tunePenalty(self, testX, testY, minPenalty=0.00001, maxPenalty=0.1, iterations=5):
        for i in range(iterations):
            costMinPenalty = self.getPenaltyCost(testX, testY, minPenalty)
            costMaxPenalty = self.getPenaltyCost(testX, testY, maxPenalty)
            if costMinPenalty < costMaxPenalty:
                maxPenalty = np.mean([minPenalty, maxPenalty])
            else:
                minPenalty = np.mean([minPenalty, maxPenalty])

        optimalPenalty = np.mean([minPenalty, maxPenalty])

        helper = NeuralNetworkHelper(self.trainX, self.trainY, penalty=optimalPenalty)
        output = helper.analyze(testX)
        plt.plot(abs(output - testY), label='analysis error')
        plt.legend(loc='upper right')
        plt.show()

# Demo
# -----------------------------------

def heartDemo():
    data = []

    FILE_NAME = 'heart-data.csv'
    with open(FILE_NAME,'r+') as csvfile:
        csvreader = csv.DictReader(csvfile)
        for row in csvreader:
            data.append(row)

    size = len(data) / 2 - 1

    trainX = []
    trainY = []
    testX = []
    testY = []

    keys = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 
        'slope', 'ca', 'thal', 'num']

    for i in range(len(data) // 2):
        item = data[i]
        entry = []
        for key in keys:
            entry.append(item[key])
        if all(attr != '?' for attr in entry):
            trainX.append(entry[:-1])
            trainY.append([entry[-1]])

        item = data[i + len(data) // 2]
        entry = []
        for key in keys:
            entry.append(item[key])
        if all(attr != '?' for attr in entry):
            testX.append(entry[:-1])
            testY.append([entry[-1]])

    trainX = np.array(trainX, dtype=float)
    trainY = np.array(trainY, dtype=float)
    testX = np.array(testX, dtype=float)
    testY = np.array(testY, dtype=float)
    
    helper = NeuralNetworkHelper(trainX, trainY)
    helper.tunePenalty(testX, testY)

heartDemo()
