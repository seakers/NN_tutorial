import csv
import json
import os
import numpy as np


def load_csv_data(filename):
    raw_array = []
    with open(os.path.join('./data', filename), newline='') as csvfile:
        csv_data = csv.reader(csvfile)
        for row in csv_data:
            raw_array.append(row)
    return raw_array


def add_pareto_information(raw_data, pareto_info_json):
    with open(os.path.join('data', pareto_info_json), newline='') as jsonfile:
        pareto_info = json.load(jsonfile)
        i = 0
        for element_info in pareto_info:
            while i < element_info["id"]:
                raw_data[i].append(0)
                i += 1
            raw_data[i].append(1)
            i += 1
        while i < len(raw_data):
            raw_data[i].append(0)
            i += 1
    return raw_data


def remove_duplicates(raw_data):
    new_data = []
    bit_strings = set()
    for row in raw_data:
        if row[0] not in bit_strings:
            new_data.append(row)
            bit_strings.add(row[0])
    return new_data


def boolean_string2boolean_array(boolean_string):
    return [b == "1" for b in boolean_string]


def boolean_array2double_array(boolean_array):
    return [float(b) for b in boolean_array]


def generate_numpy_arrays(cleaned_array, x_index, y_index):
    x = np.array([boolean_array2double_array(boolean_string2boolean_array(cleaned_row[x_index])) for cleaned_row in cleaned_array])
    y = np.array([float(cleaned_row[y_index]) for cleaned_row in cleaned_array])
    return x, y


def data_split_train_test(x, y, sample_frac):
    # Randomly shuffle data
    np.random.seed(10)
    shuffle_indices = np.random.permutation(np.arange(len(y)))
    x_shuffled = x[shuffle_indices]
    y_shuffled = y[shuffle_indices]

    # Split train/test set
    dev_sample_index = -1 * int(sample_frac * float(len(y)))
    x_train, x_test = x_shuffled[:dev_sample_index], x_shuffled[dev_sample_index:]
    y_train, y_test = y_shuffled[:dev_sample_index], y_shuffled[dev_sample_index:]

    return x_train, y_train, x_test, y_test


def score_predictor_data():
    raw_array = load_csv_data('EOSS_data.csv')
    cleaned_array = remove_duplicates(raw_array)
    x, y = generate_numpy_arrays(cleaned_array, 0, 1)
    return data_split_train_test(x, y, 0.1)


def pareto_classifier_data():
    raw_array = load_csv_data('EOSS_data.csv')
    pareto_array = add_pareto_information(raw_array, 'pareto_front.json')
    cleaned_array = remove_duplicates(pareto_array)
    x, y = generate_numpy_arrays(cleaned_array, 0, 3)
    return data_split_train_test(x, y, 0.1)
