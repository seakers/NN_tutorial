from keras.models import Sequential
from keras.layers import Dense, Dropout

pareto_classifier_perceptron = Sequential()
pareto_classifier_perceptron.add(Dense(1, input_dim=1))

#pareto_classifier_perceptron.compile(loss='', optimizer='', metrics=['accuracy'])
