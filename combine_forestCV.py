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
"""
	Created a method for calculating a random forest and returning its predicted result.
	The reason is to create seperate models for different columns. In this case, 
			
			Since 'count' = 'casual' + 'registered'
	
	We create one model to predict 'casual', and another to predict 'registered'.
	We then combine the scores of casual + registered to give us count.

"""
def calculateForestModel(col_pred, cols_remove):

	remove_columns =['bcount'] + cols_remove

	train_data, train_id = clean_data_to_numbers('data/train.csv',remove_columns)
	test_data, test_id = clean_data_to_numbers('data/test.csv')

	index_count = np.where(train_data.columns.values == col_pred)[0][0]

	"""
	*** Create the random forest object which will include all the parameters for the fit
	"""

	tuned_parameters = [{'n_estimators' : [500], 'max_features': ['auto']}]

	forest = RandomForestRegressor(n_estimators = 500, max_features='auto')

	forestcv = GridSearchCV(forest, tuned_parameters, cv=10, scoring=rmsle_scorer, n_jobs = 4, verbose=3)

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
	
	return result, test_id

arrCasualCount, _id= calculateForestModel('casual', ['registered'])

arrRegisteredCount, _id = calculateForestModel('registered', ['casual'])

arrResult = arrRegisteredCount + arrCasualCount

write_model("data/CombineRandomForestModel.csv", arrResult, _id)
pdb.set_trace()
print "done!"
