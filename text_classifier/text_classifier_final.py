from keras.layers import Input, Dense, Dropout, Embedding, Conv1D, MaxPooling1D, Flatten, concatenate
from keras.models import Model
import numpy as np


def text_classifier_model(vocab_size, maxlen, num_classes):
    nb_filters = 50
    filter_lenghts = [3, 4, 5]
    convs = []
    emb_dim = 128

    # This returns a tensor
    inputs = Input(shape=(maxlen,), dtype=np.int32)

    # a layer instance is callable on a tensor, and returns a tensor
    x = Embedding(vocab_size, emb_dim)(inputs)

    for i in filter_lenghts:
        conv = Conv1D(nb_filters, i, activation='relu')(x)
        conv = MaxPooling1D(maxlen - i + 1)(conv)
        convs.append(conv)

    x = concatenate(convs)
    x = Flatten()(x)
    x = Dropout(0.5)(x)
    predictions = Dense(num_classes, activation='softmax')(x)

    # This creates a model
    model = Model(inputs=inputs, outputs=predictions)
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model
