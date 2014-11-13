from data_helpers import *

# Import the random forest package
from sklearn.grid_search  import GridSearchCV 
from sklearn.svm import SVC
from sklearn import preprocessing	
from sklearn.cross_validation import train_test_split

tuned_parameters = [{'kernel': ['linear'], 'C':[1,10,100,1000]},
                    {'kernel': ['poly'],'degree':[2,3,4,5,6],'C': [1, 10, 100, 1000]},
                    {'kernel': ['rbf'], 'gamma': [1e-2,1e-1, 1, 1e1], 'C': [1, 10, 100, 1000]}]

remove_columns =['Age*Class', 'Fare', 'Parch']

train_data, train_passenger_id = clean_data_to_numbers('data/train.csv', remove_columns)

# for i,x in enumerate(list_of_columns)

test_data, test_passenger_id = clean_data_to_numbers('data/test.csv', remove_columns)

# If normalizeData is true then Normalize
pdb.set_trace()
X_train= preprocessing.normalize(train_data[0::,1::])
X_test = preprocessing.normalize(test_data)

Y_train = train_data[0::,0]

clf = GridSearchCV(SVC(C=1), tuned_parameters, cv=10, scoring='accuracy', n_jobs = 4, verbose=0)

clf.fit(X_train, Y_train)

result = clf.predict(X_test)

write_model("data/SVCModel.csv", result, test_passenger_id)