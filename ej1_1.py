from kohonen import Kohonen
from sklearn.preprocessing import StandardScaler
from config import load_kohonen_config

import numpy as np
import os
from matplotlib import cm
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects

from utils.parser import parse_csv_file

df = parse_csv_file('./inputs/europe.csv')
labels = df["Country"].to_numpy()
df.drop(columns=["Country"], axis=1, inplace=True)
cells = list(df.columns)
inputs = StandardScaler().fit_transform(df.values)

config = load_kohonen_config()

grid_dimension = int(config['grid_dimension'])
radius = int(config['radius'])
learning_rate = float(config['learning_rate'])
epochs = int(config['epochs'])
random_weights = config['random_weights']

kohonen = None

def count_plot():
    fig, ax = plt.subplots(figsize=(10, 10))
    heatmap = np.zeros((grid_dimension, grid_dimension))
    texts = [[[] for x in np.arange(kohonen.grid_dimension)] for y in np.arange(kohonen.grid_dimension)]

    for i in np.arange(len(inputs)):
        x, y = kohonen.find_best_neuron(inputs[i])
        heatmap[y][x] += 1
        texts[y][x].append(labels[i])
        # plt.text(x - 0.25, y, labels[i])

    for y in np.arange(kohonen.grid_dimension):
        for x in np.arange(kohonen.grid_dimension):
            country_amount = len(texts[y][x])
            for i in np.arange(country_amount):
                txt = plt.text(x - 0.35, y + (country_amount / 2) * 0.1 - i * 0.1, texts[y][x][i], color='#fff', size='large', fontweight='bold')
                txt.set_path_effects([PathEffects.withStroke(linewidth=1, foreground='#000')])
            # plt.text(x - 0.2, y, texts[y][x])

    plt.imshow(heatmap, cmap='inferno')
    plt.rc('font', size=10)
    plt.colorbar()
    plt.xticks(np.arange(grid_dimension))
    plt.yticks(np.arange(grid_dimension))
    plt.show()

# plot the amount of inputs that each neuron has won in a kxk heatmap
def count_plot_k(k):
    fig, ax = plt.subplots(figsize=(10, 10))
    heatmap = np.zeros((k, k))

    for i in np.arange(len(inputs)):
        x, y = kohonen.find_best_neuron(inputs[i])
        heatmap[y][x] += 1

    plt.imshow(heatmap, cmap='inferno')
    plt.title("Amount of inputs that each neuron has won")
    plt.colorbar()
    plt.xticks(np.arange(k))
    plt.yticks(np.arange(k))
    plt.rc('font', size=14)
    output_dir = './outputs'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    plt.savefig(f'./outputs/europe_{k}.png')
    

# plot the average value of each variable for each neuron
def average_variable_plot():
    count_matrix = np.zeros((grid_dimension, grid_dimension))
    variables_matrix = np.zeros((len(cells), grid_dimension, grid_dimension))

    for input in inputs:
        x, y = kohonen.find_best_neuron(input)
        count_matrix[y][x] += 1
        for var in range(len(cells)):
            variables_matrix[var][y][x] += kohonen.neurons[y][x].weights[var]
        
    for var in range(len(cells)):
        variables_matrix[var] /= count_matrix
    
    fig, axes = plt.subplots(2, 4, figsize=(12, 12))
    for i in range(len(cells)):
        axes[i//4][i%4].imshow(variables_matrix[i], cmap='inferno')
        axes[i//4][i%4].set_title(cells[i])
        axes[i//4][i%4].set_xticks(np.arange(grid_dimension))
        axes[i//4][i%4].set_yticks(np.arange(grid_dimension))
    
    fig.delaxes(axes[1][3])
    plt.show()

# plot the average distance to neighbours for each neuron in the grid
def matrix_plot():
    fig, ax = plt.subplots(figsize=(10, 10))
    heatmap = np.zeros((grid_dimension, grid_dimension))
    for x in range(grid_dimension):
        for y in range(grid_dimension):
            locals_weights = kohonen.neurons[y][x].weights
            average_neighbour_dist = 0
            valid_neighbours = 0
            for neighbour_x in range(x-1, x+2):
                for neighbour_y in range(y-1, y+2):
                    if neighbour_x >= 0 and neighbour_x < grid_dimension and neighbour_y >= 0 and neighbour_y < grid_dimension:
                        average_neighbour_dist += kohonen.neurons[neighbour_y][neighbour_x].distance(locals_weights)
                        valid_neighbours += 1

            heatmap[y][x] = average_neighbour_dist / valid_neighbours

    plt.imshow(heatmap, cmap= cm.gray)
    plt.title("Average distance to neighbours")
    plt.colorbar()
    plt.show()


for i in range(2, 9, 2):
    kohonen = Kohonen(i, radius, learning_rate, epochs, random_weights)
    kohonen.train(inputs)
    count_plot_k(i)

kohonen = Kohonen(grid_dimension, radius, learning_rate, epochs, random_weights)
kohonen.train(inputs)

count_plot()
matrix_plot()
average_variable_plot()

