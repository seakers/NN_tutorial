from keras.models import load_model
import os
import data_cleaning as dc
import numpy as np
import pickle
from keras.preprocessing.sequence import pad_sequences

print('Welcome to the NN predictor!\n', '1: Score Predictor\n', '2: Pareto Classifier (Perceptron)\n',
      '3: Pareto Classifier (CNN)\n', '4: Text Classifier')
number = int(input('Choose a model to use as predictor (1-4): '))

if number < 1 or number > 4:
    print('Please choose a valid option!')
    exit()

if number == 1:
    score_predictor = load_model(os.path.join('./models', 'score_predictor.h5'))
    bit_string = input("Please introduce a bit string for testing (length=60): ")
    bit_array = dc.boolean_array2double_array(dc.boolean_string2boolean_array(bit_string))
    input_array = np.array([bit_array])
    predicted_score = score_predictor.predict(input_array)
    print("The predicted science benefit for this architecture is: %.3f" % predicted_score, ". Am I close?")
elif number == 2:
    pareto_classifier = load_model(os.path.join('./models', 'pareto_classifier_perceptron.h5'))
    bit_string = input("Please introduce a bit string for testing (length=60): ")
    bit_array = dc.boolean_array2double_array(dc.boolean_string2boolean_array(bit_string))
    input_array = np.array([bit_array])
    is_pareto = round(pareto_classifier.predict(input_array)[0, 0])
    if is_pareto:
        print("I think this architecture is in the pareto front. Am I right?")
    else:
        print("I think this architecture is not in the pareto front. Am I right?")
elif number == 3:
    pareto_classifier = load_model(os.path.join('./models', 'pareto_classifier_cnn.h5'))
    bit_string = input("Please introduce a bit string for testing (length=60): ")
    bit_array = dc.boolean_array2double_array(dc.boolean_string2boolean_array(bit_string))
    input_array = dc.generate_matrices(np.array([bit_array]))
    is_pareto = round(pareto_classifier.predict(input_array)[0, 0])
    if is_pareto:
        print("I think this architecture is in the pareto front. Am I right?")
    else:
        print("I think this architecture is not in the pareto front. Am I right?")
elif number == 4:
    text_classifier = load_model(os.path.join('./models', 'text_classifier.h5'))
    text = input("Please introduce a sentence for testing: ")
    cleaned_question = dc.clean_str(text)
    # Load tokenizer
    with open(os.path.join('./models', 'tokenizer.pickle'), 'rb') as handle:
        tokenizer = pickle.load(handle)
        max_document_length = pickle.load(handle)
    # Map data into vocabulary
    x_test = np.array(tokenizer.texts_to_sequences([cleaned_question]))
    x_test = pad_sequences(x_test, maxlen=max_document_length)
    result = text_classifier.predict(x_test, verbose=True)
    classification = np.argmax(result)
    print("I think the class for this sentence is %i. Am I right?" % classification)
