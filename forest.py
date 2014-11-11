from main_2 import *

# Import the random forest package
from sklearn.ensemble import RandomForestClassifier 

import csv as csv

train_data, train_passenger_id = clean_data_to_numbers('data/train.csv')


test_data, test_passenger_id = clean_data_to_numbers('data/test.csv')

# Create the random forest object which will include all the parameters
# for the fit
forest = RandomForestClassifier(n_estimators = 10)

# Fit the training data to the Survived labels and create the decision trees

forest = forest.fit(train_data[0::,1::],train_data[0::,0])

# Take the same decision trees and run it on the test data
result = forest.predict(test_data)


write_model("data/RandomForestModel.csv", result, test_passenger_id)


# Do score here.
print "Done!"

