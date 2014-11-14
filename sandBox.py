from data_helpers import *

# Import the random forest package
from sklearn.cross_validation import train_test_split
from sklearn.grid_search  import GridSearchCV 
from sklearn.metrics import metrics as metrics
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn import preprocessing	
from sklearn import linear_model
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.feature_selection import RFECV
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
# Set the parameters by cross-validation
# 
# Columns to remove
remove_columns = ['Fare', 'Parch', 'SibSp']

tuned_parameters_1 = {'C':[1,10,100,1000],'class_weight':'auto' }
tuned_parameters_2 = [{'kernel': ['linear'], 'C':[1,10,100,1000],'gamma': [1e-3,1e-2,1e-1, 1, 5],'class_weight':['auto'] },
                    {'kernel': ['rbf'], 'gamma': [1e-3,1e-2,1e-1, 1,5], 'C': [1, 10, 100, 1000], 'class_weight':['auto']}]

train_data, train_passenger_id = clean_data_to_numbers('data/train.csv',remove_columns,normalize=True)



X_train, X_test, Y_train, Y_test =train_test_split(train_data[0::,1::],train_data[0::,0], test_size=0.333, random_state=0)
pdb.set_trace()

X_train = preprocessing.normalize(X_train)
X_test = preprocessing.normalize(X_test)

svc = LinearSVC(C=100, class_weight='auto')


rfecv = RFECV(estimator=svc, step=1, cv=5,
              scoring='accuracy')

gscv = GridSearchCV(SVC(C=1), tuned_parameters_2, cv=10, scoring='accuracy', n_jobs = 4, verbose=0)
clf = Pipeline([
  ('feature_selection', rfecv),
  ('classification', gscv)
])

clf.fit(X_train, Y_train)
pdb.set_trace()
print("Optimal number of features : %d" % rfecv.n_features_)

# Plot number of features VS. cross-validation scores
import matplotlib.pyplot as plt
plt.figure()
plt.xlabel("Number of features selected")
plt.ylabel("Cross validation score (nb of correct classifications)")
plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
plt.show()
print clf.score(X_test, Y_test)

Y_true, Y_pred = Y_test, clf.predict(X_test)

print classification_report(Y_true, Y_pred)



