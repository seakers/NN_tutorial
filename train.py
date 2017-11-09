import score_predictor.score_predictor as sp
import pareto_classifier.pareto_classifier_perceptron as pcp
import pareto_classifier.pareto_classifier_cnn as pcc
import text_classifier.text_classifier as tc
import data_cleaning

print("Welcome to the NN trainer!\n", "1: Score Predictor\n", "2: Pareto Classifier (Perceptron)\n",
      "3: Pareto Classifier (CNN)\n", "4: Text Classifier")
number = int(input("Choose a model to train (1-4): "))

if number < 1 or number > 4:
    print("Please choose a valid option!")
    exit()

if number == 1:
    pass
elif number == 2:
    x_train, y_train, x_test, y_test = data_cleaning.pareto_classifier_data()
    pcp.pareto_classifier_perceptron.fit(x_train, y_train, epochs=40, batch_size=128)
    score = pcp.pareto_classifier_perceptron.evaluate(x_test, y_test, batch_size=128)
    print("Test loss: %.2f Test accuracy: %.2f" % (score[0], score[1]))
elif number == 3:
    pass
elif number == 4:
    pass