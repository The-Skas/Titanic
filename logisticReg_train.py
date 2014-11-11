from data_helpers import *

# Import the random forest package
from sklearn.linear_model import LogisticRegression 
import csv as csv


for i, C in enumerate(10. ** np.arange(1, 4)):
	print C
	print "*********************************"
	clf_l1_LR = LogisticRegression(C=C, penalty='l1', tol=0.01)
	clf_l2_LR = LogisticRegression(C=C, penalty='l2', tol=0.01)

	print "-----------l1-{}----------------".format(C)
	print feature_selection_model(clf_l1_LR)
	print "-----------l2-{}----------------".format(C)
	print feature_selection_model(clf_l2_LR)

# Do score here.
print "Done!"

