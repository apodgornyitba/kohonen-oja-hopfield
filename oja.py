# Oja rule for PCA learning algorithm

import numpy as np


class OjaPerceptron:
    def __init__(self, training_set , eta):
        self.training_set = np.array(training_set)
        self.eta = eta

    def train(self, epochs):

        # initialize weights to random values (between 0 and 1)
        self.weights = np.random.rand(self.training_set.shape[1])

        for i in range(epochs):
            for input in self.training_set:
                y = np.dot(self.weights, input)
                self.weights += self.eta * y * (input - y * self.weights)

        return self.weights



