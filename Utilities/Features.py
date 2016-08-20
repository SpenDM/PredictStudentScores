import re
from Utilities.Globals import *


def get_features(responses):
    features = []

    for response in responses:
        response_feats = {}

        # Add features for response
        ngram_features(response, response_feats)

        features.append(response_feats)

    return features


def tokenize(response):
    processed_grams = []

    # lowercase
    response = response.lower()

    # remove punctuation
    no_punctuation_response = re.sub("\W", " ", response)

    grams = no_punctuation_response.split()
    for gram in grams:
        if gram:
            # Compress into word classes
            if gram.isdigit():
                processed_grams.append(NUMBER)
            else:
                processed_grams.append(gram)

    return processed_grams


def ngram_features(response, response_feats):
    grams = tokenize(response)
    previous_gram = ""
    two_back_gram = ""

    for gram in grams:
        # Unigrams
        response_feats[gram] = True

        # Bigrams
        if previous_gram:
            response_feats[previous_gram + "_" + gram] = True

        '''
        # Trigrams
        if two_back_gram:
            response_feats[two_back_gram + "_" + previous_gram + "_" + gram] = True
        '''

        two_back_gram = previous_gram
        previous_gram = gram
