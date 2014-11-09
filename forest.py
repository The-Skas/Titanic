from main_2 import *

# Import the random forest package
from sklearn.ensemble import RandomForestClassifier 

import csv as csv

train_data, train_passenger_id = clean_data_to_numbers('data/train.csv')


test_data, test_passenger_id = clean_data_to_numbers('data/test.csv')

# Create the random forest object which will include all the parameters
# for the fit
forest = RandomForestClassifier(n_estimators = 100)

# Fit the training data to the Survived labels and create the decision trees

forest = forest.fit(train_data[0::,1::],train_data[0::,0])

# Take the same decision trees and run it on the test data
output = forest.predict(test_data)

test_data_id = get_array_id_from_file('data/test.csv')



prediction_file = open("data/RandomForestModel.csv", "wb")
prediction_file_object = csv.writer(prediction_file)
prediction_file_object.writerow(["PassengerId", "Survived"])

for i,x in enumerate(test_passenger_id):       # For each row in test.csv
        prediction_file_object.writerow([x, output[i]])    # predict 1

prediction_file.close()
print "Done!"