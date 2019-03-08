# -*- coding: utf-8 -*-
"""
Created on Thu Feb 28 13:55:37 2019

@author: dselva
"""
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, Flatten
import matplotlib.pyplot as plt
from data_cleaning import score_predictor_data, score_predictor_cnn_data, load_csv_data
from keras.utils import plot_model

#pareto_classifier_perceptron.compile(loss='', optimizer='', metrics=['accuracy'])

## Score predictior script

# Visualize data
raw_array = load_csv_data('EOSS_data.csv')
tt = list(zip(*raw_array))
archs = tt[0]
sci = tt[1]
cos = tt[2]
plt.scatter(sci,cos)
x_train, y_train, x_test, y_test = score_predictor_data()

# Try simple pareto classifier with Perceptron
score_predictor1 = Sequential()
score_predictor1.add(Dense(1, activation='linear', input_dim=60))
score_predictor1.compile(loss='mean_squared_error', optimizer='adam')
score_predictor1.fit(x_train, y_train, epochs=40, batch_size=128)
score = score_predictor1.evaluate(x_test, y_test, batch_size=128)
print("Test MSE: %.5f" % score)
#plot_model(score_predictor1, to_file='score_predictor1.png')

# Change activation function
# See https://keras.io/activations/
score_predictor2 = Sequential()
score_predictor2.add(Dense(1, activation='relu', input_dim=60))
score_predictor2.compile(loss='mean_squared_error', optimizer='adam')
score_predictor2.fit(x_train, y_train, epochs=40, batch_size=128)
score = score_predictor2.evaluate(x_test, y_test, batch_size=128)
print("Test MSE: %.5f" % score)
#plot_model(score_predictor2, to_file='score_predictor2.png')


# Show effect of more layers
score_predictor3 = Sequential()
score_predictor3.add(Dense(200, activation='relu', input_dim=60))
score_predictor3.add(Dense(200, activation='relu'))
score_predictor3.add(Dense(200, activation='relu'))
score_predictor3.add(Dense(1))
score_predictor3.compile(loss='mean_squared_error', optimizer='adam')
score_predictor3.fit(x_train, y_train, epochs=40, batch_size=128)
score = score_predictor3.evaluate(x_test, y_test, batch_size=128)
print("Test MSE: %.5f" % score)
#plot_model(score_predictor3, to_file='score_predictor3.png')


# Show effect of regularization
score_predictor4 = Sequential()
score_predictor4.add(Dense(200, activation='relu', input_dim=60))
score_predictor4.add(Dropout(0.5))
score_predictor4.add(Dense(200, activation='relu'))
score_predictor4.add(Dropout(0.5))
score_predictor4.add(Dense(200, activation='relu'))
score_predictor4.add(Dropout(0.5))
score_predictor4.add(Dense(1))
score_predictor4.compile(loss='mean_squared_error', optimizer='adam')
score_predictor4.fit(x_train, y_train, epochs=40, batch_size=128)
score = score_predictor4.evaluate(x_test, y_test, batch_size=128)
print("Test MSE: %.5f" % score)
#plot_model(score_predictor4, to_file='score_predictor4.png')


# Show effect of different types of layers (convolutional)
x_train, y_train, x_test, y_test = score_predictor_cnn_data()
score_predictor5 = Sequential()
score_predictor5.add(Conv2D(50, (2, 5), activation='relu', input_shape=(5, 12, 1)))
score_predictor5.add(Dropout(0.25))
score_predictor5.add(Flatten())
score_predictor5.add(Dense(200, activation='relu'))
score_predictor5.add(Dropout(0.5))
score_predictor5.add(Dense(200, activation='relu'))
score_predictor5.add(Dropout(0.5))
score_predictor5.add(Dense(1))
score_predictor5.compile(loss='mean_squared_error', optimizer='adam')
score_predictor5.fit(x_train, y_train, epochs=40, batch_size=128)
score = score_predictor5.evaluate(x_test, y_test, batch_size=128)
print("Test MSE: %.5f" % score)
#plot_model(score_predictor5,to_file = 'score_predictor5.png')

