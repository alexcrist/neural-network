import numpy as np
import csv
from model.helper import NeuralNetworkHelper
from model.loader import CSVLoader

def demo():
    FILE_PATH = '../data/concrete.csv'
    OUTPUT_KEYS = ['Concrete compressive strength(MPa, megapascals) ']
    loader = CSVLoader(FILE_PATH, OUTPUT_KEYS, shuffle=True)
    data = loader.getTestTrainData()

    trainX = np.array(data['trainInputData'], dtype=float)
    trainY = np.array(data['trainOutputData'], dtype=float)

    testX = np.array(data['testInputData'], dtype=float)
    testY = np.array(data['testOutputData'], dtype=float)
    
    helper = NeuralNetworkHelper(trainX, trainY)
    helper.visualize(testX, testY)

demo()
