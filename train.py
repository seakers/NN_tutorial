import score_predictor.score_predictor_final as sp
import pareto_classifier.pareto_classifier_perceptron as pcp
import pareto_classifier.pareto_classifier_cnn as pcc
import text_classifier.text_classifier as tc
import data_cleaning
import os

print('Welcome to the NN trainer!\n', '1: Score Predictor\n', '2: Pareto Classifier (Perceptron)\n',
      '3: Pareto Classifier (CNN)\n', '4: Text Classifier')
number = int(input('Choose a model to train (1-4): '))

if number < 1 or number > 4:
    print('Please choose a valid option!')
    exit()

if not os.path.exists('models'):
    os.makedirs('models')

if number == 1:
    x_train, y_train, x_test, y_test = data_cleaning.score_predictor_data()
    sp.score_predictor.fit(x_train, y_train, epochs=40, batch_size=128)
    score = sp.score_predictor.evaluate(x_test, y_test, batch_size=128)
    print("Test MSE: %.5f" % score)
    sp.score_predictor.save(os.path.join('./models', 'score_predictor.h5'))
elif number == 2:
    x_train, y_train, x_test, y_test = data_cleaning.pareto_classifier_data(True)
    pcp.pareto_classifier_perceptron.fit(x_train, y_train, epochs=40, batch_size=128)
    score = pcp.pareto_classifier_perceptron.evaluate(x_test, y_test, batch_size=128)
    print("Test loss: %.5f Test accuracy: %.5f" % (score[0], score[1]))
    pcp.pareto_classifier_perceptron.save(os.path.join('./models', 'pareto_classifier_perceptron.h5'))
elif number == 3:
    x_train, y_train, x_test, y_test = data_cleaning.pareto_classifier_cnn_data(True)
    pcc.pareto_classifier_cnn.fit(x_train, y_train, epochs=40, batch_size=128)
    score = pcc.pareto_classifier_cnn.evaluate(x_test, y_test, batch_size=128)
    print("Test loss: %.5f Test accuracy: %.5f" % (score[0], score[1]))
    pcc.pareto_classifier_cnn.save(os.path.join('./models', 'pareto_classifier_cnn.h5'))
elif number == 4:
    x_train, y_train, x_test, y_test, vocab_size = data_cleaning.text_classifier_data()
    text_classifier = tc.text_classifier_model(vocab_size, len(x_train[0]), len(y_train[0]))
    text_classifier.fit(x_train, y_train, epochs=40, batch_size=128)
    score = text_classifier.evaluate(x_test, y_test, batch_size=128)
    print("Test loss: %.5f Test accuracy: %.5f" % (score[0], score[1]))
    text_classifier.save(os.path.join('./models', 'text_classifier.h5'))
