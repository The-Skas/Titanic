from data_helpers import *

# Import the random forest package
from sklearn.linear_model import LogisticRegression 
from sklearn.grid_search  import GridSearchCV 
from sklearn.metrics import metrics as metrics


param_grid = [
	  {'C': [1, 10, 100, 1000]}
	 ]

print "*********************************"

remove_columns = ['AgeIsNull', 'Age*Class', 'Fare', 'Embarked']

train_data, train_passenger_id = clean_data_to_numbers('data/train.csv', remove_columns)

# for i,x in enumerate(list_of_columns)

test_data, test_passenger_id = clean_data_to_numbers('data/test.csv', remove_columns)


clf = GridSearchCV(LogisticRegression(C = 1, penalty = 'l2'),param_grid, cv = 5)

clf.fit(train_data[0::,1::],train_data[0::,0])

result = clf.predict(test_data)

write_model("data/LogisticRegressionModelCV.csv", result, test_passenger_id)

pdb.set_trace()
