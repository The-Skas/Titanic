from data_helpers import *

# Import the random forest package
from sklearn.ensemble import RandomForestRegressor
from sklearn.grid_search  import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
import random
import csv as csv
# train_test_split(train_data.values[0::,1::],train_data.values[0::,0], test_size=0.01)

remove_additional_columns = ['windspeed','holiday','day']
remove_columns =['casual', 'registered']

train_data, train_id = clean_data_to_numbers('data/train.csv',remove_columns + remove_additional_columns)
test_data, test_id = clean_data_to_numbers('data/test.csv', remove_additional_columns)

index_count = np.where(train_data.columns.values == 'bcount')[0][0]

"""
*** Create the random forest object which will include all the parameters for the fit
"""

tuned_parameters = [{'n_estimators' : [500], 'max_features': ['auto'], 'n_jobs': [4]}]

forest = RandomForestRegressor(n_estimators = 500, max_features='auto', n_jobs = 4)

forestcv = GridSearchCV(forest, tuned_parameters, cv=10, scoring=rmsle_scorer, n_jobs = 4, verbose=0)

model = Pipeline([
	('regression', forestcv),
	])
# Remove the 'bcount' columns for the X value
train_X = np.delete(train_data.values, np.s_[index_count], 1)

# Use the 'bcount' columns only for the Y value
train_Y = train_data.values[0::, index_count]

# forest.fit is just for debugging
forest.fit(train_X, train_Y)

model.fit(train_X, train_Y)

result = model.predict(test_data.values)
pdb.set_trace()

write_model("data/CVRandomForestModel.csv", result, test_id)
print forestcv.best_score_
print forest.feature_importances_
print test_data.columns.values
for i,x in enumerate(test_data.columns.values):
	print "----------"
	print x
	print forest.feature_importances_[i]
pdb.set_trace()
print "Done!"

