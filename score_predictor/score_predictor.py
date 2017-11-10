from keras.models import Sequential
from keras.layers import Dense, Dropout

score_predictor = Sequential()
score_predictor.add(Dense(200, activation='relu', input_dim=60))
score_predictor.add(Dropout(0.5))
score_predictor.add(Dense(200, activation='relu'))
score_predictor.add(Dropout(0.5))
score_predictor.add(Dense(1))

score_predictor.compile(loss='mean_squared_error', optimizer='adam')
