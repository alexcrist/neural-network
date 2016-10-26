import numpy as np
import csv
from model.helper import NeuralNetworkHelper

def demo():
    data = []

    FILE_NAME = '../data/heart-data.csv'
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

demo()
