from scipy import optimize

class Trainer(object):
    ''' A trainer used to train a neural network to a dataset.

        Attributes: network - a neural network model '''

    # Creates a neural network trainer
    def __init__(self, network):
        self.network = network

    # A wrapper method needed for optimize.minimize in the train method
    def costWrapper(self, weightArray, X, Y):
        self.network.setWeights(weightArray)
        cost = self.network.cost(X, Y)
        gradient = self.network.computeGradient(X, Y)
        return cost, gradient

    # Train the network using the given input and output data
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