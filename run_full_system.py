import load_data, train_model, predict_scores
from Utilities.Globals import labeled_data_file

print("Loading Data")
load_data.main(labeled_data_file)

print("Training regression model")
train_model.main()

print("Classifying scores")
predict_scores.main()
