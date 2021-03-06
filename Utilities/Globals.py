# Data Loading
response_tag = r"(\d+)//"
scores_tag = r"scores//"
labeled_data_file = r"Data\responses_and_scores.txt"
unlabeled_data_file = r"Data\student_responses.txt"

train_test_ratio = 4  # 4:1, aka 80% train, 20% test
train_response_file = r"Data\train_responses.p"
train_score_file = r"Data\train_scores.p"
test_response_file = r"Data\test_responses.p"
test_score_file = r"Data\test_scores.p"
response_file = r"Data\responses.p"

# Training
regression_model_file = r"Model\model.m"
feature_map_file = r"Model\feature_map.m"

# Features
TOTAL_LENGTH = "TOTAL_LEN"

NUMBER = "NUMBER"

# Results
evaluation_file = r"Results\evaluation.txt"
predictions_file = r"Results\predictions.txt"
max_score = 5
