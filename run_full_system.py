import load_data, train_model, classify_scores
from Utilities.Globals import data_file

print("Loading Data")
load_data.main(data_file)

print("Training regression model")
train_model.main()

print("Classifying scores")
classify_scores.main()
