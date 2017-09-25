*--In Development--*

# PredictStudentScores
Linear regression for predicting scores for student free-text responses. Uses uni/bigrams and response length as features

# Running the program
```
run_full_system.py 
```

This will load the default file with labeled data, train a regression model on ~80% of it, and evaluate its performance (simple accuracies) on the remaining percent.

```
predict_scores.py
```

This will load the default file with unlabeled data and a previously trained model, then output its predictions.
