from math import hypot
from typing import Tuple

import numpy as np
import random

class Neuron:

    def __init__(self, weights: np.ndarray) -> None:
        self.weights = weights

    # Euclidean distance between the neuron's weights and the input
    def distance(self, x: np.ndarray):
        if len(self.weights) != len(x):
            raise Exception("Dimensions don't match")
        return np.linalg.norm(x - self.weights)
    
class Kohonen:

    def __init__(self, grid_dimension, radius, learning_rate, epochs, random_weights):
        self.neurons = [[None for _ in range(grid_dimension)] for _ in range(grid_dimension)]
        self.grid_dimension = grid_dimension
        self.radius = radius
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.random_weights = random_weights

    def train(self, inputs: list[list[float]]) -> None:
        input_amount = len(inputs) - 1
        for x in range(self.grid_dimension):
            for y in range(self.grid_dimension):
                # if flag is set, randomize weights between -1 and 1
                if self.random_weights:
                    self.neurons[x][y] = Neuron(weights=np.random.rand(len(inputs[0])) * 2 - 1)
                # else, randomize weights from the input list of values (this will decrese the amount of dead neurons)    
                else:
                    value_index = random.randint(0, input_amount)
                    rand_input = inputs[value_index]
                    self.neurons[x][y] = Neuron(weights=list(rand_input))

        total_iterations = self.epochs * len(inputs)

        iteration = 0
        r = self.radius
        # n = self.learning_rate
        n = 1
        while iteration < total_iterations:
            
            # TODO: Randomize input order, now its in order of the list

            i = iteration % len(inputs)
            rand_input = inputs[i]

            best_x, best_y = self.find_best_neuron(rand_input)

            # for x in range(best_x - r, best_x + r + 1, 1):
            #     for y in range(best_y - r, best_y + r + 1, 1):
            #         if x >= 0 and x < self.grid_dimension and y >= 0 and y < self.grid_dimension and hypot(x - best_x, y - best_y) <= r:
            #             self.neurons[y][x].weights += n * (rand_input - self.neurons[y][x].weights)

            # Iterate over all neurons and check if they are in the neighbourhood of the best neuron and update their weights
            for x in range(self.grid_dimension):
                for y in range(self.grid_dimension):
                    if hypot(abs(x - best_x), abs(y - best_y)) <= r:
                        self.neurons[x][y].weights += n * (rand_input - self.neurons[x][y].weights)
        
            iteration += 1
            if iteration % input_amount == 0:
                # n = 1 / (iteration + 1)
                n = self.learning_rate * ((total_iterations - iteration) / total_iterations)
                r = (1-self.radius) * (iteration/total_iterations) + self.radius
            # n = (0.7 - self.learning_rate)/total_iterations * iteration + self.learning_rate

    def find_best_neuron(self, input: list[float]) -> Tuple[int, int]:
        best_distance = np.inf
        best_coords = None
        for x in range(self.grid_dimension):
            for y in range(self.grid_dimension):
                distance = self.neurons[x][y].distance(input)
                if distance < best_distance:
                    best_distance = distance
                    best_coords = [x,y]
        return best_coords
