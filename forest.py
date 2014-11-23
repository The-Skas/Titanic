from data_helpers import *

# Import the random forest package
from sklearn.ensemble import RandomForestClassifier 
from sklearn.grid_search  import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import random
import csv as csv

remove_columns =['Staff','SibSp', 'Parch', 'Embarked']

train_data, train_passenger_id = clean_data_to_numbers('data/train.csv',remove_columns)


test_data, test_passenger_id = clean_data_to_numbers('data/test.csv', remove_columns)


# Create the random forest object which will include all the parameters
# for the fit
tuned_parameters = [{'n_estimators' : [2000]}]
					
random_num = 110
forest = RandomForestClassifier(n_estimators=2000,max_features=4, min_samples_split=3, random_state=47)
forestcv = GridSearchCV(forest, tuned_parameters, cv=10, scoring='accuracy', n_jobs = 4, verbose=3)


X_train, X_test, Y_train, Y_test =train_test_split(train_data.values[0::,1::],train_data.values[0::,0], test_size=0.01)

X_train_norm= preprocessing.normalize(X_train)
X_test_norm = preprocessing.normalize(X_test)

forestcv.fit(X_train_norm, Y_train)

Y_true,Y_pred =Y_test, forestcv.predict(X_test_norm)

print(accuracy_score(Y_true, Y_pred))
pdb.set_trace()

forest.fit(X_train, Y_train)

Y_true, Y_pred =Y_test , forest.predict(X_test)

# A function that transforms the given rows back to dfFN, dfFP, dfTP, dfTN

truePositives, falsePositives, trueNegatives, falseNegatives = getDataFrameConfusionMatrix(Y_pred, Y_test, X_test, test_data)
print(classification_report(Y_true, Y_pred))
print(confusion_matrix(Y_true, Y_pred))
print(accuracy_score(Y_true, Y_pred))

# Fit the training data to the Survived labels and create the decision trees
forest.fit(train_data.values[0::,1::],train_data.values[0::,0])

# Take the same decision trees and run it on the test data
result = forest.predict(test_data.values)

write_model("data/RandomForestModel.csv", result, test_passenger_id)

pdb.set_trace()

# Do score here.
print "Done!"

