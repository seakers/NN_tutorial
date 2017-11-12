from keras.models import Sequential
from keras.layers import Dense, Dropout

pareto_classifier_perceptron = Sequential()
pareto_classifier_perceptron.add(Dense(128, activation='relu', input_dim=60))
pareto_classifier_perceptron.add(Dropout(0.5))
pareto_classifier_perceptron.add(Dense(128, activation='relu'))
pareto_classifier_perceptron.add(Dropout(0.5))
pareto_classifier_perceptron.add(Dense(1, activation='sigmoid'))

pareto_classifier_perceptron.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
