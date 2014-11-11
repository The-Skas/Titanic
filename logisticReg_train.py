from data_helpers import *

# Import the random forest package
from sklearn.linear_model import LogisticRegression 
from sklearn.grid_search  import GridSearchCV 
from sklearn.metrics import metrics as metrics






for i, C in enumerate(10. ** np.arange(1, 4)):
	print C
	print "*********************************"
	clf_l1_LR = LogisticRegression(C=C, penalty='l1', tol=0.01)
	clf_l2_LR = LogisticRegression(C=C, penalty='l2', tol=0.01)

	print "-----------l1-{}----------------".format(C)
	print feature_selection_model(clf_l1_LR)
	print "-----------l2-{}----------------".format(C)
	print feature_selection_model(clf_l2_LR)
	


	

	# clf_1 = GridSearchCV(LogisticRegression(C = 1.0, penalty = 'l1'),param_grid,score_func = metrics.zero_one_loss, cv = 10)

pdb.set_trace()
print evaluate_accuracy_of_removed_columns(clf, ['Age', 'Fare', 'AgeIsNull', 'Parch', 'SibSp'])
# Do score here.
print "Done!"

