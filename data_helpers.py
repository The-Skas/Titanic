import pandas as pd
import numpy as np
import pdb
import math
import pylab as P
import csv as csv
import datetime
from sklearn import preprocessing	
from sklearn.cross_validation import train_test_split
from sklearn.metrics import metrics as met
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.feature_selection import RFECV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import make_scorer
from sklearn.metrics import mean_squared_error
from sklearn.grid_search  import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import random
import csv as csv
import matplotlib.pyplot as plt

# Globals
DROP_COL = ['']
def clean_data_to_numbers(file,additional_columns = [], normalize = False, drop_columns_default = [], doDropColumns=True):
	df = pd.read_csv(file, header=0)


	# Split datetime to get the hour.
	df['hour'] = df.datetime.map(lambda x: int(x.split(" ")[1].split(":")[0]))

	# Split datetime to get the day
	df['day'] =  df.datetime.map(lambda x: int(x.split(" ")[0].split('-')[2]))

	# Split to get month
	df['month'] = df.datetime.map(lambda x: int(x.split(" ")[0].split('-')[1]))
	
	# Split to get year
	df['year'] = df.datetime.map(lambda x: int(x.split(" ")[0].split('-')[0]))

	# The day of the week. (0 -> 6 (monday->Sunday))
	df['weekday'] = df.datetime.map(lambda x: datetime.datetime(getYear(x), getMonth(x), getDay(x)).weekday())

	# Due to more people registering per a year, combine month*year
	# df['month*year'] = df['year'].map(str) + df['month'].map(str)
	# df['month*year'] = df['month*year'].map(int)

	# Set rush-hour times. (When people go to work, leave work)
	df['rushhour'] = 0
	df.loc[(df.hour == 17) | (df.hour == 18) | (df.hour == 8),'rushhour'] = 1
	# To store Id
	_id = df['datetime']

	# Drop Id since output format issues
	if(doDropColumns):
		df = df.drop(['datetime'] +additional_columns , axis = 1)

	values = df

	return values, _id

def clean_data_to_numbers_registered(file,additional_columns = [], normalize = False, drop_columns_default = []):
	df, _id = clean_data_to_numbers(file, additional_columns, doDropColumns=False)

	# Set weather to 1 if 3. These values differ for registered vs casuals
	# df['badweather'] = 0
	# df.loc[(df.weather == 3), 'badweather'] = 1

	df = df.drop(['datetime'] +additional_columns , axis = 1)

	return df, _id


def clean_data_to_numbers_casual(file,additional_columns = [], normalize = False, drop_columns_default = []):
	df, _id = clean_data_to_numbers(file, additional_columns,doDropColumns=False)

	# df['badweather'] = 0
	# df.loc[(df.weather == 3) | (df.weather == 4) , 'badweather'] = 1
	
	df = df.drop(['datetime'] +additional_columns , axis = 1)

	return df, _id
def parseDate(str, index):
	return int(str.split(" ")[0].split('-')[index])
def getDay(str):
	return parseDate(str, 2)
def getMonth(str):
	return parseDate(str, 1)
def getYear(str):
	return parseDate(str, 0)

"""Creates the model to predict specific column in the data."""	
def fit_model_prediction_for_column(model, fileTrain='data/train.csv', fileTest='data/test.csv', Predictcolumn='Age', dropColumnsTrain=['Survived'] + DROP_COL,dropColumnsTest=DROP_COL):
	df_train = pd.read_csv(fileTrain, header=0)

	df_test  = pd.read_csv(fileTest, header=0)

	clean_up_some_values(df_train)

	clean_up_some_values(df_test)

	df_train = df_train.drop(dropColumnsTrain,axis=1)
	df_test=df_test.drop(dropColumnsTest ,axis = 1)
	
	# Get data for a specific column
	df_train_good=df_train[df_train[Predictcolumn].isnull() == False]
	# Drop all Non null indices to get all null
	df_train_null=df_train.drop(df_train_good.index)
	
	df_test_good=df_test[df_test[Predictcolumn].isnull() == False]

	df_test_null=df_test.drop(df_test_good.index)

	# Joins up all Good data from train and test
	df_train = pd.concat([df_train_good, df_test_good])
	# Joins up all Null data to predict
	df_test = pd.concat([df_train_null,  df_test_null])
	# We drop 'Age' or whatever column since we have no more use.
	df_test=df_test.drop(Predictcolumn ,axis = 1)

	
 
	train_data = df_train.values
	test_data = df_test.values

	# the column that we will predict
	indexColumn = np.where(df_train.columns.values == Predictcolumn)[0][0]
	# Column to predict
	train_withColRemoved = np.delete(train_data,np.s_[indexColumn], 1)
	# Split
	X_train, X_test, Y_train, Y_test = train_test_split(train_withColRemoved,train_data[0::,indexColumn], test_size=0.33, random_state=0)

	model.fit(X_train, Y_train)

 	Y_true, Y_pred = Y_test,model.predict(X_test)

 	print met.explained_variance_score(Y_true, Y_pred)
 	print met.mean_absolute_error(Y_true, Y_pred)
	print "stop"

	return model


"""
	Outputs To file
	fileName: The name of the file to output.
	output: An array of results which are 1 or 0.
"""
def write_model(fileName, output, id):
	prediction_file = open(fileName, "wb")
	prediction_file_object = csv.writer(prediction_file)
	prediction_file_object.writerow(["datetime", "count"])
	
	for i,x in enumerate(id):       # For each row in test.csv
	        prediction_file_object.writerow([x, output[i].astype(int)])    # predict 1

	prediction_file.close()

def evaluate_accuracy_of_removed_columns(model,columns=[], normalizeData = False):
	train_data, train_passenger_id = clean_data_to_numbers('data/train.csv', columns)
	"""
	This function returns the accuracy of the model given the columns to be removed 
	"""
	# for i,x in enumerate(list_of_columns)

	train_data, train_passenger_id = clean_data_to_numbers('data/train.csv', columns)

	X_train, X_test, Y_train, Y_test =train_test_split(train_data[0::,1::],train_data[0::,0], test_size=0.5, random_state=0)

	# If normalizeData is true then Normalize
	if(normalizeData == True):
		X_train= preprocessing.normalize(X_train)
		X_test = preprocessing.normalize(X_test)

	# Create the random forest object which will include all the parameters
	# for the fit

	# Fit the training data to the Survived labels and create the decision trees
	model.fit(X_train,Y_train)
	
	return model.score(X_test, Y_test)

def feature_selection_model(col_pred, cols_remove):

	# Backward Feature Selection
	result,_id,best_accuracy,df = calculateForestModel(col_pred=col_pred, cols_remove=cols_remove,casual=False)
	
	final_removeable_columns = list()
	
	mutable_list =  list(final_removeable_columns)
	
	array_of_best_results = list()

	amount_redundant_loop = 0
	for i, x in enumerate(final_removeable_columns):
		print(i)
		print(x),
		best_column_i = -1
		for j, y in enumerate(mutable_list):
			temp_removeable_columns = list(final_removeable_columns)

			# We pick the index from mutable list, and not list_of_columns
			# Although in the first iteration they have the same values,
			# After each x+1, mutable list will have a column removed from it.
			
			temp_removeable_columns.append(mutable_list[j])
			result, _id, temp_accuracy, df = calculateForestModel(col_pred=col_pred, cols_remove=cols_remove,additional_cols_remove=temp_removeable_columns)

			# if The accuracy improved after removing the given columns
			if(temp_accuracy > best_accuracy):
				best_accuracy = temp_accuracy
				best_column_i = j

				col_acc_list = list()
				col_acc_list.append(best_accuracy)
				col_acc_list.append(temp_removeable_columns)

				array_of_best_results.append(col_acc_list)
			
			print(j),
		# Add the column that provided the most accuracy when removed.
		if(best_column_i != -1):
			final_removeable_columns.append(mutable_list[best_column_i])
			del mutable_list[best_column_i]
		else:
			++amount_redundant_loop


		# Note: If best_column is empty, then dont add anything to final_removable_clumns
		# 
	# TO SORT  l.sort(key=lambda x: x[2])
	
	array_of_best_results.sort(key=lambda x: x[0])

	# Reverse array output to match less, is better
	return array_of_best_results[::-1]




def getDataFrameConfusionMatrix(Y_pred, Y_test, X_test, df):
	truePositives, falsePositives = list(), list()
	trueNegatives, falseNegatives = list(), list()
	for i, vali in enumerate(Y_pred):
		if(vali == 1):
			# Correct
			if(Y_pred[i] == Y_test[i]):
				truePositives.append(X_test[i])
			else:
				falsePositives.append(X_test[i])
		elif(vali == 0):
			if(Y_pred[i] == Y_test[i]):
				trueNegatives.append(X_test[i])
			else:
				falseNegatives.append(X_test[i])

	return (pd.DataFrame(truePositives, columns=df.columns.values), 
		   pd.DataFrame(falsePositives, columns=df.columns.values), 
		   pd.DataFrame(trueNegatives, columns=df.columns.values), 
		   pd.DataFrame(falseNegatives, columns=df.columns.values))

""" 
	'calculateForestModel(col_pred, cols_remove, additional_cols_remove, casual)'

	Created a method for calculating a random forest and returning its predicted result.
	The reason is to create seperate models for different columns. In this case, 
			
			Since 'count' = 'casual' + 'registered'
	
	We create one model to predict 'casual', and another to predict 'registered'.
	We then combine the scores of casual + registered to give us count.

	returns: result, id

"""

def calculateForestModel(col_pred, cols_remove, casual=True, additional_cols_remove=[]):

	remove_columns =['bcount'] + cols_remove

	train_data, train_id = 0 , 0
	test_data, test_id = 0, 0

	if(col_pred == 'casual'):
		train_data, train_id = clean_data_to_numbers_casual('data/train.csv',remove_columns + additional_cols_remove)
		test_data, test_id = clean_data_to_numbers_casual('data/test.csv', additional_cols_remove)
	else:
		train_data, train_id = clean_data_to_numbers_registered('data/train.csv',remove_columns + additional_cols_remove)
		test_data, test_id = clean_data_to_numbers_registered('data/test.csv', additional_cols_remove)
			
		

	index_count = np.where(train_data.columns.values == col_pred)[0][0]

	"""
	*** Create the random forest object which will include all the parameters for the fit
	"""

	tuned_parameters = [{'n_estimators' : [1], 'max_features': ['auto'],'n_jobs':[1]}]

	forest = RandomForestRegressor(n_estimators = 1, max_features='auto', n_jobs=1)

	forestcv = GridSearchCV(forest, tuned_parameters, cv=10, scoring=rmsle_scorer, n_jobs = 1, verbose=3)

	model = forestcv
	# Remove the 'bcount' columns for the X value
	train_X = np.delete(train_data.values, np.s_[index_count], 1)

	# Use the 'bcount' columns only for the Y value
	train_Y = train_data.values[0::, index_count]

	# forest.fit is just for debugging
	forest.fit(train_X, train_Y)

	model.fit(train_X, train_Y)
	pdb.set_trace()
	result = model.predict(test_data.values)
	
	print forestcv.best_score_
	print test_data.columns.values + forest.feature_importances_.astype(np.str)
	return result, test_id, forestcv.best_score_, test_data
# Calculates the Root mean squared Log error
def rmsle(Y_true, Y_pred):
	return math.sqrt(mean_squared_error(np.log(Y_true+1), np.log(Y_pred+1)))

rmsle_scorer = make_scorer(rmsle, greater_is_better=False)




