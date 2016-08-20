import _pickle
import sys

import numpy as np
from sklearn import linear_model
from sklearn.externals import joblib
from sklearn.feature_extraction import DictVectorizer

from Utilities.Globals import *
from Utilities import Features


def main():
    # get data
    responses, scores = load_train_data()

    if responses and scores:
        # train model
        features = Features.get_features(responses)
        classifier, feature_map = train_regression_model(features, scores)

        # Write models to file
        joblib.dump(classifier, regression_model_file)
        _pickle.dump(feature_map, open(feature_map_file, "wb"))
    else:
        sys.stderr.write("Error: no labeled data found.")


def train_regression_model(feature_dicts, labels):
    # Convert Data to vectors
    sent_vectors, labels_for_classifier, feature_map = vectorize_train_data(feature_dicts, labels)

    # Create Model
    classifier = linear_model.LinearRegression()
    classifier.fit(sent_vectors, labels_for_classifier)

    return classifier, feature_map


def vectorize_train_data(feature_dicts, labels):
    # convert to vectors
    dict_vec = DictVectorizer()
    sentence_vectors = dict_vec.fit_transform(feature_dicts).toarray()

    # map features to the appropriate index in the established vector representation
    feature_names = dict_vec.get_feature_names()
    feature_map = {}
    for index, feat in enumerate(feature_names):
        feature_map[feat] = index

    return sentence_vectors, np.array(labels), feature_map


def load_train_data():
    responses = []
    scores = []

    try:
        responses = _pickle.load(open(train_response_file, "rb"))
        scores = _pickle.load(open(train_score_file, "rb"))
    except IOError:
        sys.stderr.write("Can't open training data files. Make sure you have run load_data.py\n")

    return responses, scores


if __name__ == '__main__':
    main()
