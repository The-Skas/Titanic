import pandas as pd
import numpy as np
import pdb
import math
import pylab as P
import csv as csv
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
# Globals
DROP_COL = ['Name','Sex', 'PassengerId','Ticket' ,'Cabin']
le_Cabin = 0;
le_Ticket = 0;
def clean_data_to_numbers(file,additional_columns = [], normalize = False, drop_columns_default = []):
	df = pd.read_csv(file, header=0)

	# Split datetime to get the hour.
	df['time'] = df.datetime.map(lambda x: int(x.split(" ")[1].split(":")[0]))

	# Split datetime to get the day
	df['day'] =  df.datetime.map(lambda x: int(x.split(" ")[0].split('-')[2]))

	# Split to get month
	df['month']= df.datetime.map(lambda x: int(x.split(" ")[0].split('-')[1]))
	
	# To store Id
	_id = df['datetime']

	# Drop Id since output format issues
	df = df.drop(['datetime'] +additional_columns , axis = 1)

	values = df

	return values, _id

d

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

def feature_selection_model(model, normalizeData=False):
	list_of_columns =  ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'Gender', 'AgeIsNull', 'FamilySize', 'Staff', 'Prefix']

	mutable_list = list(list_of_columns)

	final_removeable_columns = list()
	# Backward Feature Selection
	best_accuracy = evaluate_accuracy_of_removed_columns(model,[],normalizeData)


	array_of_best_results = list()

	amount_redundant_loop = 0
	for i, x in enumerate(list_of_columns):
		print(i)
		print(x),
		best_column_i = -1
		for j, y in enumerate(mutable_list):
			temp_removeable_columns = list(final_removeable_columns)

			# We pick the index from mutable list, and not list_of_columns
			# Although in the first iteration they have the same values,
			# After each x+1, mutable list will have a column removed from it.
			
			temp_removeable_columns.append(mutable_list[j])
			temp_accuracy = evaluate_accuracy_of_removed_columns(model,temp_removeable_columns,normalizeData)

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

	return array_of_best_results




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

# Calculates the Root mean squared Log error
def rmsle(Y_true, Y_pred):
	return math.sqrt(mean_squared_error(np.log(Y_true+1), np.log(Y_pred+1)))

rmsle_scorer = make_scorer(rmsle, greater_is_better=False)




