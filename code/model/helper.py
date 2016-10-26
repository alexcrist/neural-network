import numpy as np
import matplotlib.pyplot as plt
from model.neural import NeuralNetwork
from model.trainer import Trainer

class NeuralNetworkHelper(object):
    ''' A helper class that builds a trained network.

        Attributes: network - a neural network
                    trainX  - training input data
                    trainY  - training output data
                    xMax    - max value of training input data
                    yMax    - max value of training output data '''

    # Creates a neural network helper
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

    # Gets the cost of a given penalty and a set of input and output data
    def getPenaltyCost(self, testX, testY, penalty):
        costSum = 0
        trials = 10
        for i in range(trials):
            network = NeuralNetworkHelper(self.trainX, self.trainY, penalty=penalty)
            output = network.analyze(testX)
            costSum += np.sum(abs(output - testY))
        return costSum / trials

    # Tunes the optimal penalty
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