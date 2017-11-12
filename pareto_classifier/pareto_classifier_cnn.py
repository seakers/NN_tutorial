from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten

pareto_classifier_cnn = Sequential()
pareto_classifier_cnn.add(Conv2D(50, (2, 5), activation='relu', input_shape=(5, 12, 1)))
#pareto_classifier_cnn.add(MaxPooling2D(pool_size=(2, 2)))
pareto_classifier_cnn.add(Dropout(0.25))
pareto_classifier_cnn.add(Flatten())
pareto_classifier_cnn.add(Dense(200, activation='relu'))
pareto_classifier_cnn.add(Dropout(0.5))
pareto_classifier_cnn.add(Dense(1, activation='sigmoid'))

pareto_classifier_cnn.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
