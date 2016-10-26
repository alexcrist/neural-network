import numpy as np
import csv
import random

class CSVLoader(object):
    '''Contains common operations for loading and processing CSV data files.

       Attributes: data       - a list of dictionary entries representing 
                                numerical data
                   outputKeys - list of keys corrseponding to the output columns 
                                in the data'''

    # Creates a CSVLoader
    def __init__(self, filePath, outputKeys, shuffle=False):
        self.data = self.loadDataFromPath(filePath)
        self.outputKeys = outputKeys
        self.inputKeys = self.getInputKeys()
        if shuffle:
            random.shuffle(self.data)

    # Returns a list of dictionary entries from a path to CSV file
    def loadDataFromPath(self, filePath):
        data = []
        with open(filePath,'r+') as csvfile:
            csvreader = csv.DictReader(csvfile)
            for row in csvreader:
                data.append(row)
        return data

    # Returns a list of the input keys for the data
    def getInputKeys(self):
        inputKeys = []
        entry = self.data[0]
        for key in entry:
            if key not in self.outputKeys:
                inputKeys.append(key)
        return inputKeys

    # Returns a dictionary with input and output data, each value formatted as a 
    # list of lists
    def getInputOutputData(self):
        inputData = []
        outputData = []
        for entry in self.data:
            inputEntry = []
            outputEntry = []

            for key in self.inputKeys:
                value = entry[key]
                inputEntry.append(value)

            for key in self.outputKeys:
                value = entry[key]
                outputEntry.append(value)

            inputData.append(inputEntry)
            outputData.append(outputEntry)

        data = {}
        data['inputData'] = inputData
        data['outputData'] = outputData
        return data

    # Divides the data into two halves for testing and training. Returns a
    # dictionary containing all the data entries
    def getTestTrainData(self):
        inputOutputData = self.getInputOutputData()
        inputData = inputOutputData['inputData']
        outputData = inputOutputData['outputData']

        length = len(inputData)
        halfway = length // 2

        data = {}
        data['testInputData'] = inputData[0:halfway]
        data['trainInputData'] = inputData[halfway:length]
        data['testOutputData'] = outputData[0:halfway]
        data['trainOutputData'] = outputData[halfway:length]
        return data


    



