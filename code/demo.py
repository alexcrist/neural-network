import numpy as np
from model.helper import NeuralNetworkHelper
from model.loader import CSVLoader

def demo():
    # Constants for the file path and the output keys
    FILE_PATH = '../data/concrete.csv'
    OUTPUT_KEYS = ['Concrete compressive strength(MPa, megapascals) ']

    # Load testing and training data from the CSV file
    loader = CSVLoader(FILE_PATH, OUTPUT_KEYS, shuffle=True)
    data = loader.getTestTrainData()

    # Training data (X = input, Y = output)
    trainX = np.array(data['trainInputData'], dtype=float)
    trainY = np.array(data['trainOutputData'], dtype=float)

    # Testing data (X = input, Y = output)
    testX = np.array(data['testInputData'], dtype=float)
    testY = np.array(data['testOutputData'], dtype=float)

    # Create and train a new neural network
    helper = NeuralNetworkHelper(trainX, trainY, penalty=.0000001)

    # Test and visualize the networks calculations
    helper.visualize(testX, testY)

demo()
