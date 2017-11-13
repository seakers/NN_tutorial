from keras.layers import Input, Dense, Dropout, Embedding, Conv1D, MaxPooling1D, Flatten, concatenate
from keras.models import Model
import numpy as np


def text_classifier_model(vocab_size, maxlen, num_classes):
    # This returns a tensor
    inputs = Input(shape=(maxlen,), dtype=np.int32)
    predictions = Dense(num_classes, activation='')(inputs)

    # This creates a model
    model = Model(inputs=inputs, outputs=predictions)
#    model.compile(optimizer='',
#                  loss='',
#                  metrics=['accuracy'])

    return model
