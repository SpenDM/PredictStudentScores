import _pickle
import sys
from collections import defaultdict

import numpy as np
from sklearn.externals import joblib

from Utilities.Globals import *
from Utilities import Features


def main():
    # get data
    responses, scores = load_test_data()

    # load model
    model, feature_map = load_regression_model()

    # predict scores with model
    predictions = predict_scores(responses, model, feature_map)

    # round predictions to the nearest whole number
    rounded_predictions = [round(prediction) for prediction in predictions]

    # output scores

    # evaluate
    evaluate(responses, scores, rounded_predictions)


def predict_scores(responses, model, feature_map):
    features_per_response = Features.get_features(responses)

    number_of_responses = len(features_per_response)
    number_of_features = len(feature_map)

    # Vectorize sentences and get correct data format
    test_vectors = [vectorize_feature_set(feature_set, feature_map) for feature_set in features_per_response]
    test_array = np.reshape(test_vectors, (number_of_responses, number_of_features))

    # Predict using model
    predictions = model.predict(test_array)

    return predictions


def vectorize_feature_set(feats, feature_map):
    """ Convert features to a vector where feature positions match with those found in training """
    vector = [0 for _ in range(len(feature_map))]
    grams = feats.keys()
    for gram in grams:
        if gram in feature_map:
            index = feature_map[gram]
            vector[index] = 1
    return vector


def evaluate(responses, scores, predictions):
    correct = 0
    off_by_n = defaultdict(int)
    correct_predictions = []
    mistakes = []

    for prediction, score, response in zip(predictions, scores, responses):
        if prediction == score:
            # Record correct predictions
            correct += 1
            correct_prediction = (prediction, score, response)
            correct_predictions.append(correct_prediction)
        else:
            # Record errors
            difference = abs(prediction - score)
            off_by_n[difference] += 1
            mistake = (prediction, score, response)
            mistakes.append(mistake)

    find_and_output_evaluation(correct, correct_predictions, off_by_n, mistakes)


def find_and_output_evaluation(correct, correct_predictions, off_by_n, mistakes):
    # Find accuracy
    accuracies = calculate_accuracies(correct, off_by_n)

    # Output results
    out_file = open(evaluation_file, "w")

    # Exactly correct
    out_file.write("Correct: " + str(correct) + "\n")
    out_file.write("Accuracy: " + str(accuracies[0]) + "\n\n")

    # Off by n
    for n in range(1, max_score):
        correct += off_by_n[n]
        out_file.write("Off by " + str(n) + ": " + str(off_by_n[n]) + "\n")
        out_file.write("Off by " + str(n) + " or less: " + str(correct) + "\n")
        out_file.write("Accuracy: " + str(accuracies[n]) + "\n\n")

    # Output specific predictions
    out_file.write("Mistakes:\n")
    output_predictions(mistakes, out_file)

    out_file.write("\nCorrect:\n")
    output_predictions(correct_predictions, out_file)


def calculate_accuracies(correct, off_by_n):
    accuracies = defaultdict(int)
    total_wrong = sum(off_by_n.values())

    # Exact accuracy
    accuracies[0] = calculate_accuracy(correct, total_wrong)

    # Accuracy for each threshold of difference
    for n in range(1, max_score):
        correct += off_by_n[n]
        total_wrong -= off_by_n[n]

        accuracies[n] = calculate_accuracy(correct, total_wrong)

    return accuracies


def calculate_accuracy(correct, total_wrong):
    accuracy = 0
    if correct:
        accuracy = float(correct) / float(correct + total_wrong)
    return accuracy


def output_predictions(predictions, out_file):
    for prediction in predictions:
        out_file.write(str(prediction[0]) + " " + str(prediction[1]) + " " + str(prediction[2]) + "\n")


def load_regression_model():
    model = None
    feature_map = None

    try:
        model = joblib.load(regression_model_file)
        feature_map = _pickle.load(open(feature_map_file, "rb"))
    except IOError:
        print("Error: can't find trained model and feature map. Run \"train_model.py\" and make sure model files are "
              "in the correct location with the correct names")

    return model, feature_map


def load_test_data():
    responses = []
    scores = []

    try:
        responses = _pickle.load(open(test_response_file, "rb"))
        scores = _pickle.load(open(train_score_file, "rb"))
    except IOError:
        sys.stderr.write("Can't open training data files. Make sure you have run load_data.py\n")

    return responses, scores


if __name__ == '__main__':
    main()
