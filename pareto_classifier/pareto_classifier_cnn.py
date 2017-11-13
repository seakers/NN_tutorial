from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten

pareto_classifier_cnn = Sequential()
pareto_classifier_cnn.add(Dense(1, input_dim=1))

#pareto_classifier_cnn.compile(loss='', optimizer='', metrics=['accuracy'])
