import csv
import json
import os
import numpy as np
import spacy
import keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import pickle

nlp = spacy.load('en_core_web_sm')


def load_csv_data(filename):
    raw_array = []
    with open(os.path.join('./data', filename), newline='') as csvfile:
        csv_data = csv.reader(csvfile)
        for row in csv_data:
            raw_array.append(row)
    return raw_array


def add_pareto_information(raw_data, pareto_info_json):
    with open(os.path.join('./data', pareto_info_json), newline='') as jsonfile:
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
    return raw_data, len(pareto_info)


def clean_str(line):
    doc = nlp(line)
    # Pre-process the strings
    tokens = []
    for token in doc:

        # If stopword or punctuation, ignore token and continue
        if (token.is_stop and not (token.lemma_ == "which" or token.lemma_ == "how" or token.lemma_ == "what"
                                   or token.lemma_ == "when" or token.lemma_ == "why")) \
                or token.is_punct or token.is_space:
            continue

        # Lemmatize the token and yield
        tokens.append(token.lemma_)
    return " ".join(tokens)


def load_data_and_labels():
    # Load all the categories
    x_text = []
    labels = []

    # Add texts and labels
    files_list = ["0.txt", "1.txt", "2.txt"]
    for filename in files_list:
        label = int(filename.split('.', 1)[0])
        with open(os.path.join('./data', filename), 'r') as file:
            for line in file:
                clean_line = clean_str(line)
                # Add to general training
                x_text.append(clean_line)
                labels.append(label)

    y = np.array(labels)
    y = keras.utils.to_categorical(y, 3)
    return x_text, y


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


def generate_matrices(linear_input):
    return np.expand_dims(np.reshape(linear_input, (-1, 5, 12)), -1)

def same_pos_and_neg(x, y, num):
    shuffle_indices = np.random.permutation(np.arange(len(y)))
    counter_pos = 0
    counter_neg = 0
    trimmed_x = []
    trimmed_y = []
    for index in shuffle_indices:
        if y[index] == 0 and counter_neg < num:
            trimmed_x.append(x[index])
            trimmed_y.append(y[index])
            counter_neg += 1
        if y[index] == 1 and counter_pos < num:
            trimmed_x.append(x[index])
            trimmed_y.append(y[index])
            counter_pos += 1
    return np.array(trimmed_x), np.array(trimmed_y)

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


def score_predictor_cnn_data():
    raw_array = load_csv_data('EOSS_data.csv')
    cleaned_array = remove_duplicates(raw_array)
    x, y = generate_numpy_arrays(cleaned_array, 0, 1)
    x = generate_matrices(x)
    return data_split_train_test(x, y, 0.1)


def pareto_classifier_data(same_size):
    raw_array = load_csv_data('EOSS_data.csv')
    pareto_array, num_pos = add_pareto_information(raw_array, 'pareto_front.json')
    cleaned_array = remove_duplicates(pareto_array)
    x, y = generate_numpy_arrays(cleaned_array, 0, 3)
    if same_size:
        x, y = same_pos_and_neg(x, y, num_pos)
    return data_split_train_test(x, y, 0.1)


def pareto_classifier_cnn_data(same_size):
    raw_array = load_csv_data('EOSS_data.csv')
    pareto_array, num_pos = add_pareto_information(raw_array, 'pareto_front.json')
    cleaned_array = remove_duplicates(pareto_array)
    x, y = generate_numpy_arrays(cleaned_array, 0, 3)
    x = generate_matrices(x)
    if same_size:
        x, y = same_pos_and_neg(x, y, num_pos)
    return data_split_train_test(x, y, 0.1)


def text_classifier_data():
    text_x, y = load_data_and_labels()
    # Build vocabulary
    max_document_length = max([len(x.split(" ")) for x in text_x])
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(text_x)
    sequences = tokenizer.texts_to_sequences(text_x)
    data = pad_sequences(sequences, maxlen=max_document_length)
    index_count = len(tokenizer.word_index)
    x_train, y_train, x_test, y_test = data_split_train_test(data, y, 0.1)

    # Save Tokenizer
    with open(os.path.join('./models', 'tokenizer.pickle'), 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(max_document_length, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # Return all information
    return x_train, y_train, x_test, y_test, index_count+1
