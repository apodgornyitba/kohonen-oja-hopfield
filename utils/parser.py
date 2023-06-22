import pandas as pd
import numpy as np

def parse_csv_file(file: str):
    df = pd.read_csv(file)
    return df

def parse_combined_matrix(file_path):
    with open(file_path, 'r') as file:
        matrix_text = file.read()

    lines = matrix_text.strip().split('\n')
    matrix_dict = {}
    matrix_size = len(lines[0].split())

    for i in range(0, len(lines), matrix_size + 1):
        letter = lines[i:i+matrix_size]
        matrix = [[int(num) for num in line.split()] for line in letter]
        matrix_dict[chr(65 + i // (matrix_size + 1))] = matrix

    return matrix_dict