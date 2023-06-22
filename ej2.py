from config import load_hopfield_config
from utils.parser import parse_combined_matrix
from hopfield import Hopfield

import numpy as np
import copy
import itertools
import matplotlib.pyplot as plt


def print_n_othogonals(letters_martix, n):
    flattened_letters = []

    for letter in letters_martix:
        flattened_letters.append(np.array(letter, dtype=float).flatten())

    for subset in list(itertools.combinations(np.arange(len(flattened_letters)), n)):
        letters = []
        for letter_index in subset:
            letters.append(flattened_letters[letter_index])
        score = orthogonal_score(letters)
        # score = orthogonal_score([flattened_letters[subset[0]], flattened_letters[subset[1]], flattened_letters[subset[2]], flattened_letters[subset[3]]])
        if score < 4:
            chars = [chr(97+value) for value in subset]
            # print('({}, {}, {}, {}) are somewhat orthogonal'.format(chr(97+subset[0]), chr(97+subset[1]), chr(97+subset[2]), chr(97+subset[3])))
            print('{} are somewhat orthogonal'.format(chars))
            print('score: ' + str(score))
            print()
    #     if len(orthogonals) < n:
    #         orthogonals.append((score, subset))
    #     else:
    #         add_if_more_orthogonal(orthogonals, score)

    # return orthogonals
    
def orthogonal_score(arrays):
    num_arrays = len(arrays)
    score = 0

    for i in range(num_arrays):
        for j in range(i+1, num_arrays):
            dot_product = np.dot(arrays[i], arrays[j])
            score += abs(dot_product)

    return score


def check_orthogonal(matrix1, matrix2):
    dot_product = np.dot(matrix1.flatten(), matrix2.flatten())
    return np.isclose(dot_product, 0)

def find_orthogonal_sets(matrices):
    num_matrices = len(matrices)
    orthogonal_sets = []

    for i in range(num_matrices - 3):
        for j in range(i+1, num_matrices - 2):
            for k in range(j+1, num_matrices - 1):
                for l in range(k+1, num_matrices):
                    if (check_orthogonal(matrices[i], matrices[j]) and
                        check_orthogonal(matrices[i], matrices[k]) and
                        check_orthogonal(matrices[i], matrices[l]) and
                        check_orthogonal(matrices[j], matrices[k]) and
                        check_orthogonal(matrices[j], matrices[l]) and
                        check_orthogonal(matrices[k], matrices[l])):
                        orthogonal_sets.append((i, j, k, l))

    return orthogonal_sets


def print_matrix(matrix):
    for row in matrix:
        row_str = ' '.join(['*' if val == 1 else ' ' for val in row])
        print(row_str)

def add_noise(matrix, percentage: np.ndarray):
    flattened_matrix = matrix.flatten()  # Flatten the matrix into a 1D array
    num_elements = flattened_matrix.shape[0]
    num_noise_elements = int(num_elements * percentage)

    # Randomly select indices to add noise
    noise_indices = np.random.choice(num_elements, num_noise_elements, replace=False)

    # Invert the selected elements
    flattened_matrix[noise_indices] *= -1

    # Reshape the modified array back to a 5x5 matrix
    noisy_matrix = flattened_matrix.reshape(matrix.shape)
    return noisy_matrix


input_file = './inputs/letters.txt'

# Parse the input file
matrix_dict = parse_combined_matrix(input_file)

#load the hopfield config
config = load_hopfield_config()

# print_n_othogonals(matrix_dict.values(), 3)

# test the hopfield algorithm
hopfield = Hopfield([np.array(matrix_dict['G']).flatten(), np.array(matrix_dict['R']).flatten(), np.array(matrix_dict['T']).flatten(), np.array(matrix_dict['V']).flatten()])

noise_matrix = np.array(copy.deepcopy(matrix_dict['V']))
noise_matrix = add_noise(noise_matrix, 0.5)
# noise_matrix = np.array([[1, 1, 1, 1, 1], [-1, -1, -1, -1, -1], [1, -1, 1, 1, -1], [1, -1, -1, 1, 1], [-1, 1, -1, 1, -1]])

matrix_test = np.array(noise_matrix).flatten()

found, state, states, energies, i = hopfield.train(matrix_test)

print("Iterations: " + str(i))
print("Found: " + str(found))
print("Sent:")
print_matrix(noise_matrix)
print("Received:")
for iteration_state in states:
    print_matrix(iteration_state)
print("Energies:" + str(energies))

# Plot the energy
fig, ax = plt.subplots()
x = np.arange(i)
plt.plot(x, energies, 'o-')

plt.ylabel('Energía')
plt.xlabel('Número de iteración')

plt.xticks(x)
plt.show()


def noise_plot():
    iters_per_letter = 10
    noises = np.arange(0.05, 0.55, 0.05)
    noises_perc = noises * 100

    orth_exactitude = []
    orth_false_positives = []

    half_orth_exactitude = []
    half_orth_false_positives = []

    not_orth_exactitude = []
    not_orth_false_positives = []

    orth_hopfield = Hopfield([np.array(matrix_dict['Q']).flatten(), np.array( matrix_dict['R']).flatten(), np.array(matrix_dict['T']).flatten(), np.array(matrix_dict['V']).flatten()])
    orth_letters = np.array(['Q', 'R', 'T', 'V'])
    
    half_orth_hopfield = Hopfield([np.array(matrix_dict['O']).flatten(), np.array( matrix_dict['Q']).flatten(), np.array(matrix_dict['T']).flatten(), np.array(matrix_dict['X']).flatten()])
    half_orth_letters = np.array(['O', 'Q', 'T', 'X'])
    
    not_orth_hopfield = Hopfield([np.array(matrix_dict['A']).flatten(), np.array( matrix_dict['F']).flatten(), np.array(matrix_dict['P']).flatten(), np.array(matrix_dict['R']).flatten()])
    not_orth_letters = np.array(['A', 'F', 'P', 'R'])

    for noise in noises:
        noise_diff_perc = 100
        false_positives_perc = 0
        for letter in orth_letters:
            for i in np.arange(iters_per_letter):
                noise_matrix = copy.deepcopy(np.array(matrix_dict[letter]).flatten())
                noise_matrix = add_noise(noise_matrix, noise)
                found, state, states, energies, i = orth_hopfield.train(noise_matrix)

                if not np.array_equal(state, np.array(matrix_dict[letter])):
                    noise_diff_perc -= 100 / (iters_per_letter * len(orth_letters))
                    if found:
                        false_positives_perc += 100 / (iters_per_letter * len(orth_letters))
        orth_exactitude.append(noise_diff_perc)
        orth_false_positives.append(false_positives_perc)
        
        noise_diff_perc = 100
        false_positives_perc = 0
        for letter in half_orth_letters:
            for i in np.arange(iters_per_letter):
                noise_matrix = copy.deepcopy(np.array(matrix_dict[letter]).flatten())
                noise_matrix = add_noise(noise_matrix, noise)
                found, state, states, energies, it = half_orth_hopfield.train(noise_matrix)

                if not np.array_equal(state, np.array(matrix_dict[letter])):
                    noise_diff_perc -= 100 / (iters_per_letter * len(half_orth_letters))
                    if found:
                        false_positives_perc += 100 / (iters_per_letter * len(orth_letters))
        half_orth_exactitude.append(noise_diff_perc)
        half_orth_false_positives.append(false_positives_perc)
        
        noise_diff_perc = 100
        false_positives_perc = 0
        for letter in not_orth_letters:
            for i in np.arange(iters_per_letter):
                noise_matrix = copy.deepcopy(np.array(matrix_dict[letter]).flatten())
                noise_matrix = add_noise(noise_matrix, noise)
                found, state, states, energies, it = not_orth_hopfield.train(noise_matrix)

                if not np.array_equal(state, np.array(matrix_dict[letter])):
                    noise_diff_perc -= 100 / (iters_per_letter * len(not_orth_letters))
                    if found:
                        false_positives_perc += 100 / (iters_per_letter * len(orth_letters))
        not_orth_exactitude.append(noise_diff_perc)
        not_orth_false_positives.append(false_positives_perc)

    fig, ax = plt.subplots(2, figsize=(10, 10))
    ax[0].plot(noises_perc, orth_false_positives, 'o-', linestyle='dotted', label='Orthogonal')
    ax[0].plot(noises_perc, half_orth_false_positives, 'o-', linestyle='dotted', label='Half Orthogonal')
    ax[0].plot(noises_perc, not_orth_false_positives, 'o-', linestyle='dotted', label='Not Orthogonal')

    ax[0].set_ylabel('Falsos positivos (%)')
    ax[0].set_xlabel('Ruido (%)')

    ax[0].set_xticks(noises_perc)
    ax[0].legend()


    ax[1].plot(noises_perc, orth_exactitude, 'o-', linestyle='dotted', label='Orthogonal')
    ax[1].plot(noises_perc, half_orth_exactitude, 'o-', linestyle='dotted', label='Half Orthogonal')
    ax[1].plot(noises_perc, not_orth_exactitude, 'o-', linestyle='dotted', label='Not Orthogonal')

    ax[1].set_ylabel('Exactitud (%)')
    ax[1].set_xlabel('Ruido (%)')

    ax[1].set_xticks(noises_perc)
    ax[1].legend()

    plt.show()


def stored_patterns_amount():
    fig, ax = plt.subplots()
    x = [2, 3, 4, 5, 6, 7]
    y = []
    y.append(orthogonal_score([np.array(matrix_dict['G']).flatten(), np.array( matrix_dict['R']).flatten()]))
    y.append(orthogonal_score([np.array(matrix_dict['G']).flatten(), np.array( matrix_dict['R']).flatten(), np.array( matrix_dict['T']).flatten()]))
    y.append(orthogonal_score([np.array(matrix_dict['G']).flatten(), np.array( matrix_dict['R']).flatten(), np.array(matrix_dict['T']).flatten(), np.array(matrix_dict['V']).flatten()]))
    y.append(orthogonal_score([np.array(matrix_dict['G']).flatten(), np.array( matrix_dict['R']).flatten(), np.array(matrix_dict['T']).flatten(), np.array(matrix_dict['V']).flatten(), np.array(matrix_dict['Z']).flatten()]))
    y.append(orthogonal_score([np.array(matrix_dict['A']).flatten(), np.array( matrix_dict['J']).flatten(), np.array(matrix_dict['L']).flatten(), np.array(matrix_dict['T']).flatten(), np.array(matrix_dict['V']).flatten(), np.array(matrix_dict['X']).flatten()]))
    y.append(orthogonal_score([np.array(matrix_dict['A']).flatten(), np.array( matrix_dict['J']).flatten(), np.array(matrix_dict['L']).flatten(), np.array(matrix_dict['T']).flatten(), np.array(matrix_dict['V']).flatten(), np.array(matrix_dict['X']).flatten(), np.array(matrix_dict['R']).flatten()]))

    plt.plot(x, y, 'o-', linestyle='dotted')
    plt.vlines(3.75, ymin=0, ymax=70, color='r', linestyle='dashed', label='Límite del 15%')

    plt.ylabel('Score')
    plt.xlabel('Cantidad de patrones almacenados')
    plt.xticks(x)
    plt.legend()

    plt.show()

# noise_plot()
# stored_patterns_amount()
