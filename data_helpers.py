import pandas as pd
import numpy as np
import pdb
import math
import pylab as P
import csv as csv
from sklearn import preprocessing	
from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction import DictVectorizer

def clean_data_to_numbers(file,additional_columns = [], drop_columns_default = ['Sex', 'Name','Cabin', 'Ticket']):
	df = pd.read_csv(file, header=0)
	# Convert gender to number
	df['Gender'] = df['Sex'].map({'female': 0, 'male': 1})

	# Maps all non null values of Embarked to numbers.
	df['Embarked']=  df[df['Embarked'].isnull() == False].Embarked.map({'C':1,'Q':2,'S':3})
	# Gets the median
	Embarked_median = df['Embarked'].median()
	# Overwrites all of column 'Embarked' null values to equal the median 'Embarked'
	# TODO: Create a model to predict 'Embarked'.
	df['Embarked']=df['Embarked'].fillna(Embarked_median)

	# Clean Cabin
	df.Cabin = df.Cabin.fillna('Unknown')
	le = preprocessing.LabelEncoder()
	le.fit(df.Cabin)
	df.Cabin = le.transform(df.Cabin)


	# Clean Ticket
	df.Ticket = df.Ticket.fillna('Unknown')
	le = preprocessing.LabelEncoder()
	le.fit(df.Ticket)
	df.Ticket = le.transform(df.Ticket)


	# Creates an array of 6 values. 2 Rows, 3 columns.
	median_ages = np.zeros((2,3))

	# For each Male/Female, we will have Three different median ages
	# depending on what their Economic class ('Pclass') is.
	for i in range(0,2):
		for j in range(df['Pclass'].min(), df['Pclass'].max()+1):
			median_ages[i,j-1] = df[(df['Gender'] == i ) & (df['Pclass'] == j)].Age.dropna().median()

	# AgeIsNull
	df['AgeIsNull'] = pd.isnull(df.Age).astype(int)

	#  stores the median age for rows with null 'Age'
	for i in range(0, 2):
	    for j in range(0, 3):
	        df.loc[(df.Age.isnull()) & (df.Gender == i) & (df.Pclass == j+1),'Age'] = median_ages[i,j]

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

	df['FamilySize'] = df['SibSp'] + df['Parch']

	# This multiplies the Age of the person by the social 
	# class. It adds to the fact that higher ages are even
	# LESS likely to survive
	df['Age*Class'] = df.Age * df.Pclass

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


	return df.values, passengerIds

def get_array_id_from_file(file):
	df = pd.read_csv(file, header=0)

	return df['PassengerId']

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
	list_of_columns =  ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'Gender', 'AgeIsNull', 'FamilySize', 'Age*Class']

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