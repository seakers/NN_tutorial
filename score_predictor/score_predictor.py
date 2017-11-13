from keras.models import Sequential
from keras.layers import Dense, Dropout

score_predictor = Sequential()
score_predictor.add(Dense(1, input_dim=1))

#score_predictor.compile(loss='', optimizer='')
