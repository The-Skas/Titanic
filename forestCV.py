from data_helpers import *

# Import the random forest package
from sklearn.ensemble import RandomForestRegressor
from sklearn.grid_search  import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import random
import csv as csv


remove_columns =['casual', 'registered']

train_data, train_id = clean_data_to_numbers('data/train.csv',remove_columns)
test_data, test_id = clean_data_to_numbers('data/test.csv', [])

index_count = np.where(train_data.columns.values == 'bcount')[0][0]
# Create the random forest object which will include all the parameters
# for the fit
tuned_parameters = [{'n_estimators' : [200], 'max_features': ['auto']}]
random_num = 110
pdb.set_trace()
forest = RandomForestRegressor(n_estimators = 100, max_features='auto', verbose=3)
forestcv = GridSearchCV(forest, tuned_parameters, cv=10, scoring=rmsle_scorer, n_jobs = 4, verbose=3)
# forestcv = forest

# train_normalized= preprocessing.normalize(train_data.values)
# test_normalized = preprocessing.normalize(test_data.values)

train_X = np.delete(train_data.values, np.s_[index_count], 1)
train_Y = train_data.values[0::, index_count]

pdb.set_trace()
forestcv.fit(train_X, train_Y)

result = forestcv.predict(test_data.values)
pdb.set_trace()
write_model("data/CVRandomForestModel.csv", result, test_id)

pdb.set_trace()

# Do score here.
print "Done!"

