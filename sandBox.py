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


df = pd.read_csv(file, header=0)
df_ok = df.dropna()
df_bad = df.drop(df_ok.index)

