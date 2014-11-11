from data_helpers import *

# Import the random forest package
from sklearn.linear_model import LogisticRegression 
from sklearn.grid_search  import GridSearchCV 
from sklearn.metrics import metrics as metrics


param_grid = [
	  {'C': [1, 10, 100, 1000]}
	 ]

print "*********************************"
clf_l1_LR = GridSearchCV(LogisticRegression(C = 1, penalty = 'l1'),param_grid, cv = 5)
clf_l2_LR = GridSearchCV(LogisticRegression(C = 1, penalty = 'l2'),param_grid, cv = 5)

print "-----------l1-{}----------------"
print feature_selection_model(clf_l1_LR)
print "-----------l2-{}----------------"
print feature_selection_model(clf_l2_LR)

train_data, train_passenger_id = clean_data_to_numbers('data/train.csv', columns)
