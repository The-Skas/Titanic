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

# Globals
DROP_COL = ['Name','Sex', 'PassengerId','Ticket' ,'Cabin']
le_Cabin = 0;
le_Ticket = 0;
def clean_data_to_numbers(file,additional_columns = [], normalize = False, drop_columns_default = ['Sex', 'Name','Cabin', 'Ticket']):
	df = pd.read_csv(file, header=0)

	clean_up_some_values(df)
	X = get_X_data(df,'Age')

	gscv = RandomForestRegressor(n_estimators = 1000)
	model = Pipeline([
	  ('regression', gscv)
	])
	


	model = fit_model_prediction_for_column(model)
	y = model.predict(X)

	df['AgeIsNull'] = pd.isnull(df.Age).astype(int)

	df.Age[df.Age.isnull() == True] = y
	# Creates an array of 6 values. 2 Rows, 3 columns.
	median_ages = np.zeros((2,3))

	# For each Male/Female, we will have Three different median ages
	# depending on what their Economic class ('Pclass') is.
	for i in range(0,2):
		for j in range(df['Pclass'].min(), df['Pclass'].max()+1):
			median_ages[i,j-1] = df[(df['Gender'] == i ) & (df['Pclass'] == j)].Age.dropna().median()
	# AgeIsNull
	
	

	

	 # stores the median age for rows with null 'Age'
	for i in range(0, 2):
	    for j in range(0, 3):
	        df.loc[(df.Age.isnull()) & (df.Gender == i) & (df.Pclass == j+1),'Age'] = median_ages[i,j]
	df_null_Age = df[df['Age'].isnull() == True]

	
	# Convert all floats to a range of 0.5 or 1.0
	# The reason being to fit the compo rules (Refer to data)
	# df['Age']= df['Age'].map(lambda x: math.ceil(x * 2.0) * 0.5)

	# *** DO MEAN FOR FARE ****
	mean_fare = np.zeros((2,3))
	for i in range(0,2):
		for j in range(df['Pclass'].min(), df['Pclass'].max()+1):
			mean_fare[i,j-1] = df[(df['Gender'] == i ) & (df['Pclass'] == j)].Fare.dropna().mean()


	for i in range(0, 2):
	    for j in range(0, 3):
	        df.loc[(df.Fare == 0) & (df.Gender == i) & (df.Pclass == j+1),'Fare'] = mean_fare[i,j]
	# This creates a new column ('AgeIsNull') 
	# 
	# pd: this is the pandas library
	# pd.isnull(arg1): this is a function that converts the dataFrame rows
	# 				   into a true/false table.

	df['FamilySize'] = df['SibSp'] + df['Parch'] + 1

	# This multiplies the Age of the person by the social 
	# class. It adds to the fact that higher ages are even
	# LESS likely to survive
	
	# Since skipi doesnt work well with strings
	df.dtypes[df.dtypes.map(lambda x: x=='object')]
	# Setting up for machine learning yikes! Horrible!
	# The values you drop can improve or make worse.
	df = df.drop(drop_columns_default + additional_columns, axis=1)
	# Drops all columns that have any null value.. 
	# uh? wtf? Super bad.
	df = df.dropna()


	# To store Id
	passengerIds = df['PassengerId']

	# Drop Id since output format issues
	df = df.drop(['PassengerId'], axis = 1)

	values = df

	return values, passengerIds
def clean_up_some_values(df):
	df['Gender'] = df['Sex'].map({'female': 0, 'male': 1})

	# Maps all non null values of Embarked to numbers.
	df['Embarked']=  df[df['Embarked'].isnull() == False].Embarked.map({'C':1,'Q':2,'S':3})
	# Gets the median
	Embarked_median = df['Embarked'].median()
	# Overwrites all of column 'Embarked' null values to equal the median 'Embarked'
	# TODO: Create a model to predict 'Embarked'.
	df['Embarked']=df['Embarked'].fillna(Embarked_median)

	# Clean Cabin
	global le_Cabin
	df.Cabin = df.Cabin.fillna('Unknown')
	le_Cabin = preprocessing.LabelEncoder()
	le_Cabin.fit(df.Cabin)
	df.Cabin = le_Cabin.transform(df.Cabin)

	# Clean Ticket
	global le_Ticket
	df.Ticket = df.Ticket.fillna('Unknown')
	le_Ticket = preprocessing.LabelEncoder()
	le_Ticket.fit(df.Ticket)
	df.Ticket = le_Ticket.transform(df.Ticket)

	df.Fare=df.Fare.fillna(np.mean(df.Fare))

	# Assumes anyone with Fare 0 to be staff. Staff are more likely to die.
	df['Staff'] = 0
	df.loc[df.Fare == 0 , 'Staff'] = 1

	df.loc[df.Fare == 0, 'Fare'] = df.Fare.median()

	# pdb.set_trace()
	df['Prefix'] =  df['Name'].map( lambda x: x.split(",")[1].split(" ")[1])

	# Simplify High Survival Ladies since They All Survive.
	df.loc[df.Prefix.isin(['Mlle.', 'Mme.', 'Lady.', 'Ms.', 'the']), 'Prefix'] = 'HighWoman'

	df.loc[df.Prefix.isin(['Master.','Sir.']), 'Prefix'] = 'HighMen'

	df.loc[df.Prefix.isin(['Capt.', 'Col.' ,'Don.', 'Dr.' ,'Jonkheer.', 'Major.' ]), 'Prefix'] = 'WorkForce'
	df.loc[(df.Staff == 1),'Prefix'] = 'Rev.'

	le = preprocessing.LabelEncoder()
	le.fit(df.Prefix)
	df.Prefix = le.transform(df.Prefix)

	# Create new Column Prefix-Pclass-Gender
	print("Do Prefix-Pclass-Gender")
	# Must initialize new row as follows
	df['PclassGenderPref'] = 0	
	df.PclassGenderPref = df.Gender.map(str)+ df.Pclass.map(str) +df.Prefix.map(str)
	df.PclassGenderPref = df.PclassGenderPref.map(int)
	print("Check.")
def get_X_data(df, predictColumn, dropColumns=DROP_COL ):
	# Copy df to not alter anything.
	df_t = df.copy()

	df_t = df_t[df_t[predictColumn].isnull() == True]

	if('Survived' in df_t):
		df_t=df_t.drop('Survived', axis=1)

	df_t=df_t.drop(dropColumns + [predictColumn], axis=1)

	data=df_t.values
	

	return data


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


def get_array_id_from_file(file):
	df = pd.read_csv(file, header=0)

	return df['PassengerId']

"""
	Outputs To file
	fileName: The name of the file to output.
	output: An array of results which are 1 or 0.
"""
def write_model(fileName, output, passengersId):
	prediction_file = open(fileName, "wb")
	prediction_file_object = csv.writer(prediction_file)
	prediction_file_object.writerow(["PassengerId", "Survived"])
	
	for i,x in enumerate(passengersId):       # For each row in test.csv
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



			
		





