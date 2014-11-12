from data_helpers import *

# Import the random forest package
from sklearn.cross_validation import train_test_split
from sklearn.grid_search  import GridSearchCV 
from sklearn.metrics import metrics as metrics
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn import preprocessing	
# Set the parameters by cross-validation
# 
columns = []

tuned_parameters = [{'kernel': ['linear'], 'C':[1,10,100,1000]},
                    {'kernel': ['poly'],'degree':[2,3,4,5,6],'C': [1, 10, 100, 1000]},
                    {'kernel': ['rbf'], 'gamma': [1e-2,1e-1, 1, 1e1], 'C': [1, 10, 100, 1000]}]


print "*********************************"

train_data, train_passenger_id = clean_data_to_numbers('data/train.csv',columns)

X_train, X_test, Y_train, Y_test =train_test_split(train_data[0::,1::],train_data[0::,0], test_size=0.5, random_state=0)
# pdb.set_trace()

X_train= preprocessing.normalize(X_train)
X_test = preprocessing.normalize(X_test)





clf = GridSearchCV(SVC(C=1), tuned_parameters, cv=10, scoring='accuracy', n_jobs = 4, verbose=0)

print feature_selection_model(clf, True)


clf = GridSearchCV(SVC(C=1), tuned_parameters, cv=10, scoring='accuracy', n_jobs = 4, verbose=0)

clf.fit(X_train, Y_train)

pdb.set_trace()

Y_true, Y_pred = Y_test, clf.predict(X_test)


pdb.set_trace()

print(classification_report(Y_true, Y_pred))

(Pdb) df_ok = df.dropna()
(Pdb) df_bad = df.drop(df_ok.index)
