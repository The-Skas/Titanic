from data_helpers import *

# Import the random forest package
from sklearn.linear_model import LogisticRegression
from sklearn.grid_search  import GridSearchCV 
from sklearn.metrics import metrics as metrics
import csv as csv

remove_columns = ['Age', 'Fare', 'AgeIsNull', 'Parch', 'SibSp']
pdb.set_trace()
train_data, train_passenger_id = clean_data_to_numbers('data/train.csv', remove_columns)

# for i,x in enumerate(list_of_columns)

test_data, test_passenger_id = clean_data_to_numbers('data/test.csv', remove_columns)

# Create the random forest object which will include all the parameters
# for the fit
lR = LogisticRegression(C=100, penalty='l1', tol=0.01)

# # Fit the training data to the Survived labels and create the decision trees
lR.fit(train_data[0::,1::],train_data[0::,0])

result = lR.predict(test_data)

# write_model("data/LogisticRegressionModel.csv", result, test_passenger_id)

# Do score here.
print "Done!"

