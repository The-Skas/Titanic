from data_helpers import *

# Import the random forest package
from sklearn.ensemble import RandomForestClassifier 
from sklearn.grid_search  import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import random
import csv as csv


remove_columns =['Parch', 'SibSp', 'AgeIsNull', 'Staff', 'Embarked']

train_data, train_passenger_id = clean_data_to_numbers('data/train.csv',remove_columns)


test_data, test_passenger_id = clean_data_to_numbers('data/test.csv', remove_columns)


# Create the random forest object which will include all the parameters
# for the fit
tuned_parameters = [{'n_estimators' : [2000], 'max_features': [2,3,4,5], 'min_samples_split':[1,2,3], \
					'random_state':[int(random.random() * 100),int(random.random() * 100),int(random.random() * 100)]}]
random_num = 110
forest = RandomForestClassifier(n_estimators = 2000, max_features=3, min_samples_split=2, random_state= random_num)
forestcv = GridSearchCV(forest, tuned_parameters, cv=10, scoring='accuracy', n_jobs = 4, verbose=3)


train_normalized= preprocessing.normalize(train_data.values)
test_normalized = preprocessing.normalize(test_data.values)


forestcv.fit(train_normalized[0::, 1::], train_data.values[0::, 0])

result = forestcv.predict(test_normalized)

write_model("data/CVRandomForestModel.csv", result, test_passenger_id)

pdb.set_trace()

# Do score here.
print "Done!"

