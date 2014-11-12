from data_helpers import *

# Import the random forest package
from sklearn.cross_validation import train_test_split
from sklearn.grid_search  import GridSearchCV 
from sklearn.metrics import metrics as metrics
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn import preprocessing	
from sklearn import linear_model
# Set the parameters by cross-validation
# 
# Columns to remove
remove_columns = ['Survived']


clf = linear_model.RidgeCV(alphas=[0.1, 1.0, 10.0])

# df_Age_ok   =df[df.Age.isnull() == False]
# df_Age_null = df.drop(df_Age_ok.index)
fit_model_prediction_for_column(clf)


train_data, train_passenger_id = clean_data_to_numbers('data/test.csv',remove_columns)

X_train, X_test, Y_train, Y_test =train_test_split(train_data[0::,1::],train_data[0::,0], test_size=0.5, random_state=0)