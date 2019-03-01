# -*- coding: utf-8 -*-
"""
Created on Thu Feb 28 13:55:37 2019

@author: dselva
"""
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, Flatten
import matplotlib.pyplot as plt
from data_cleaning import pareto_classifier_data, load_csv_data
from keras.utils import plot_model

#pareto_classifier_perceptron.compile(loss='', optimizer='', metrics=['accuracy'])

## Pareto classifier script

# Visualize data
raw_array = load_csv_data('EOSS_data.csv')
tt = list(zip(*raw_array))
archs = tt[0]
sci = tt[1]
cos = tt[2]
plt.scatter(sci,cos)
x_train, y_train, x_test, y_test = pareto_classifier_data(True)

# Try simple pareto classifier with Perceptron
pareto_classifier1 = Sequential()
pareto_classifier1.add(Dense(1, activation='linear'))
pareto_classifier1.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
pareto_classifier1.fit(x_train, y_train, epochs=40, batch_size=128)
score = pareto_classifier1.evaluate(x_test, y_test, batch_size=128)
print("Test accuracy: %.5f" % (score[1]))
plot_model(pareto_classifier1,to_file = 'pareto_classifier1.png')

# Change activation function
# See https://keras.io/activations/
pareto_classifier2 = Sequential()
pareto_classifier2.add(Dense(1, activation='sigmoid'))
pareto_classifier2.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
pareto_classifier2.fit(x_train, y_train, epochs=40, batch_size=128)
score = pareto_classifier2.evaluate(x_test, y_test, batch_size=128)
print("Test accuracy: %.5f" % (score[1]))
plot_model(pareto_classifier2,to_file = 'pareto_classifier2.png')


# Show effect of more layers

pareto_classifier3 = Sequential()
pareto_classifier3.add(Dense(128, activation='relu', input_dim=60))
pareto_classifier3.add(Dense(128, activation='relu'))
pareto_classifier3.add(Dense(1, activation='sigmoid'))

pareto_classifier3.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
pareto_classifier3.fit(x_train, y_train, epochs=40, batch_size=128)
score = pareto_classifier3.evaluate(x_test, y_test, batch_size=128)
print("Test accuracy: %.5f" % (score[1]))
plot_model(pareto_classifier3,to_file = 'pareto_classifier3.png')


# Show effect of regularization

pareto_classifier4 = Sequential()
pareto_classifier4.add(Dense(128, activation='relu', input_dim=60))
pareto_classifier4.add(Dropout(0.5))
pareto_classifier4.add(Dense(128, activation='relu'))
pareto_classifier4.add(Dropout(0.5))
pareto_classifier4.add(Dense(1, activation='sigmoid'))

pareto_classifier4.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
pareto_classifier4.fit(x_train, y_train, epochs=40, batch_size=128)
score = pareto_classifier4.evaluate(x_test, y_test, batch_size=128)
print("Test accuracy: %.5f" % (score[1]))
plot_model(pareto_classifier4,to_file = 'pareto_classifier4.png')


# Show effect of different types of layers (convolutional)
pareto_classifier5 = Sequential()
pareto_classifier5.add(Conv2D(50, (2, 5), activation='relu', input_shape=(5, 12, 1)))
#pareto_classifier_cnn.add(MaxPooling2D(pool_size=(2, 2)))
pareto_classifier5.add(Dropout(0.25))
pareto_classifier5.add(Flatten())
pareto_classifier5.add(Dense(200, activation='relu'))
pareto_classifier5.add(Dropout(0.5))
pareto_classifier5.add(Dense(1, activation='sigmoid'))

pareto_classifier5.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
pareto_classifier5.fit(x_train, y_train, epochs=40, batch_size=128)
score = pareto_classifier5.evaluate(x_test, y_test, batch_size=128)
print("Test accuracy: %.5f" % (score[1]))
plot_model(pareto_classifier5,to_file = 'pareto_classifier5.png')

