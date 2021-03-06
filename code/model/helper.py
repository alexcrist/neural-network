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
    def __init__(self, trainX, trainY, penalty=0.00015):
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

    # Given testing data, graph the expected output vs. the actual given output
    def visualize(self, textX, testY):
        # Arrays for actual outputs, calculated ouptus, and the difference
        actual = testY
        calculated = self.analyze(textX)
        diff = abs(calculated - actual)

        # Sort the three arrays by the values of the actual outputs
        combined = sorted(zip(actual, calculated, diff))
        actualSorted = [a for a, c, d in combined]
        calculatedSorted = [c for a, c, d in combined]
        diffSorted = [d for a, c, d in combined]

        # Calculate the average error
        avgError = round(self.getAvgError(calculated, actual), 2)

        # Plot the data
        plt.figure(figsize=(15, 7))
        plt.plot(actualSorted, label='actual')
        plt.plot(calculatedSorted, label='calculated')
        plt.plot(diffSorted, label='difference')
        plt.plot([], 'w', label='Avg. error: ' + str(avgError) + '%')
        plt.legend()
        plt.show()

    # Returns the average error between the given datasets.
    def getAvgError(self, calculated, actual):
        sum = 0
        for i in range(len(calculated)):
            calculatedVal = calculated[i][0]
            actualVal = actual[i][0]
            percentDiff = 100 * abs(actualVal - calculatedVal) / actualVal
            sum += percentDiff
        avgError = sum / len(calculated)
        return avgError




