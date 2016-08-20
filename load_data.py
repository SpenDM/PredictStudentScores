import _pickle
import argparse
import re
import sys

from Utilities.Globals import *


def main(filename):
    # Get file lines
    data_file_lines = get_file_lines(filename)

    # Parse file
    responses, scores = parse_data_file(data_file_lines)

    # Split into train and test if there are labels
    if scores:
        split_train_and_test(responses, scores)
    # Otherwise output responses
    else:
        _pickle.dump(responses, open(response_file, "wb"))


def get_file_lines(filename):
    data_file_lines = []
    try:
        with open(filename, "r") as file:
            data_file_lines = file.readlines()
    except IOError:
        sys.stderr.write("Can't read file " + filename)
    return data_file_lines


def parse_data_file(data_file_lines):
    responses = []
    scores = []

    response = []
    in_response = False
    in_scores = False

    for end_index, line in enumerate(data_file_lines):
        line = line.rstrip('\n')

        # Once we're at the scores, grab them and stop
        if in_scores:
            scores = get_scores(line)
            break

        # Ignore blank lines
        if line:
            # Update responses and scores with info from line
            in_response, in_scores = check_for_response_line_content(line, in_response, in_scores, response, responses)

    # Add the last response if this wasn't done when scores were found
    if response and not in_scores:
        finish_previous_response_and_begin_new(response, responses)

    return responses, scores


def check_for_response_line_content(line, in_response, in_scores, response, responses):
    # Check for response tag
    found_new_response, in_response = check_for_response_tag(line, in_response, response, responses)

    if not found_new_response:
        # Check for scores tag
        in_scores, in_response = check_for_scores_tag(line, in_response, response, responses)

        # If we're in a response and haven't found any tags, continue the response
        if not in_scores and in_response:
            response.append(line)

    return in_response, in_scores


def check_for_response_tag(line, in_response, response, responses):
    found_response = re.match(response_tag, line)
    if found_response:
        if in_response:
            # if already in a response, add previous response and begin new one
            finish_previous_response_and_begin_new(response, responses)
        else:
            in_response = True

    return found_response, in_response


def check_for_scores_tag(line, in_response, response, responses):
    in_scores = False

    found_scores = re.match(scores_tag, line)
    if found_scores:
        in_scores = True

        # Finish response if scores tag is found
        if in_response:
            finish_previous_response_and_begin_new(response, responses)
            in_response = False

    return in_scores, in_response


def finish_previous_response_and_begin_new(response, responses):
    # Append finished response
    full_response = " ".join(response)
    responses.append(full_response)

    # Reset for new response
    response.clear()


def get_scores(line):
    text_scores = line.split()

    # convert strings to integers
    labels_and_scores = [int(score) for score in text_scores]

    # remove labels -- assumes scores are in order
    scores = []
    for index, number in enumerate(labels_and_scores):
        # grab every other number
        if (index % 2) == 1:
            scores.append(number)

    return scores


def even_out_responses_and_scores(responses, scores):
    # Ensure responses is no longer than scores
    if len(responses) > len(scores):
        responses = responses[:len(scores)]
        sys.stderr.write("Warning: not all responses have scores\n")

    # Ensure scores is no longer than responses
    if len(scores) > len(responses):
        scores = scores[:len(responses)]
        sys.stderr.write("Warning: there are more scores than responses\n")

    return responses, scores


def split_train_and_test(responses, scores):
    train_responses = []
    train_scores = []
    test_responses = []
    test_scores = []

    for index, (response, score) in enumerate(zip(responses, scores)):
        # Add every nth pair to test, where n is based on the ratio given
        if index % train_test_ratio == 0:
            test_responses.append(response)
            test_scores.append(score)
        # Add the rest to train
        else:
            train_responses.append(response)
            train_scores.append(score)

    output_split_data(train_responses, train_scores, train_response_file, train_score_file)
    output_split_data(test_responses, test_scores, test_response_file, test_score_file)


def output_split_data(responses, scores, response_file, score_file):
    _pickle.dump(responses, open(response_file, "wb"))
    _pickle.dump(scores, open(score_file, "wb"))


# Handle Arguments
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("data_file", help="path of file containing responses and scores")
    args = parser.parse_args()

    main(args.data_file)
