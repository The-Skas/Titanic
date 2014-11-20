from data_helpers import *

# Import the random forest package
from sklearn.cross_validation import train_test_split
from sklearn.grid_search  import GridSearchCV 
from sklearn.metrics import metrics as metrics
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn import preprocessing	
from sklearn.tree import DecisionTreeClassifier
# Set the parameters by cross-validation
# 'Age*Class', 'Fare', 'Parch'
# columns = ['Ticket', 'Cabin', 'Embarked','SibSp', 'Parch','AgeIsNull']
# columns = [ 'Parch', 'SibSp','Cabin','Ticket', 'Name','Fare', 'Pclass','Embarked', 'FamilySize','Prefix','Staff','AgeIsNull']
columns = []
tuned_parameters = [{'kernel': ['linear'], 'C':[1,10,100,1000],'gamma': [1e-3,1e-2,1e-1, 1, 5],'class_weight':['auto'] },
                    {'kernel': ['rbf'], 'gamma': [1e-3,1e-2,1e-1, 1,10,100], 'C': [1, 10, 100, 1000], 'class_weight':['auto'], 'tol':[0.1,0.01,0.001, 0.0001]}]


print "*********************************"

train_data, train_passenger_id = clean_data_to_numbers('data/train.csv',columns)
pdb.set_trace()
X_train, X_test, Y_train, Y_test =train_test_split(train_data.values[0::,1::],train_data.values[0::,0], test_size=0.33, random_state=0)
# pdb.set_trace()

X_train= preprocessing.normalize(X_train)
X_test = preprocessing.normalize(X_test)




svc = SVC(C=1)
clf = GridSearchCV(svc, tuned_parameters, cv=5, scoring='accuracy', n_jobs = 4, verbose=0)

print feature_selection_model(clf, True)

clf = GridSearchCV(SVC(C=1), tuned_parameters, cv=10, scoring='accuracy', n_jobs = 4, verbose=0)

clf.fit(X_train, Y_train)


Y_true, Y_pred = Y_test, clf.predict(X_test)


pdb.set_trace()

print(classification_report(Y_true, Y_pred))
print(confusion_matrix(Y_true, Y_pred))
