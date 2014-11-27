

import math


# Import the random forest package
from sklearn.ensemble import RandomForestRegressor
from sklearn.grid_search  import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
import random
import csv as csv

from datetime import datetime

# load training data
import graphlab as graphlab

# load training data
training_sframe = graphlab.SFrame.read_csv('data/train.csv')

# train a model
features = ['datetime', 'season', 'holiday', 'workingday', 'weather',
            'temp','atemp', 'humidity', 'windspeed']
m = graphlab.boosted_trees.create(training_sframe,
                            features=features,
                            target='bcount', objective='regression',
                            num_iterations=20)

# predict on test data
test_sframe = graphlab.SFrame.read_csv('data/test.csv')
prediction = m.predict(test_sframe)



def make_submission(prediction, filename='submission.txt'):
    with open(filename, 'w') as f:
        f.write('datetime,count\n')
        submission_strings = test_sframe['datetime'] + ',' + prediction.astype(str)
        for row in submission_strings:
            f.write(row + '\n')

make_submission(prediction, 'submission1.txt')



from datetime import datetime
date_format_str = '%Y-%m-%d %H:%M:%S'

def parse_date(date_str):
    """Return parsed datetime tuple"""
    d = datetime.strptime(date_str, date_format_str)
    return {'year': d.year, 'month': d.month, 'day': d.day,
            'hour': d.hour, 'weekday': d.weekday()}

def process_date_column(data_sframe):
    """Split the 'datetime' column of a given sframe"""
    parsed_date = data_sframe['datetime'].apply(parse_date).unpack(column_name_prefix='')
    for col in ['year', 'month', 'day', 'hour', 'weekday']:
        data_sframe[col] = parsed_date[col]

process_date_column(training_sframe)
process_date_column(test_sframe)




# Create three new columns: log-casual, log-registered, and log-count
for col in ['casual', 'registered', 'bcount']:
    training_sframe['log-' + col] = training_sframe[col].apply(lambda x: math.log(1 + x))






new_features = features + ['year', 'month', 'weekday', 'hour']
new_features.remove('datetime')

m1 = graphlab.boosted_trees.create(training_sframe,
                             features=new_features,
                             target='log-casual')

m2 = graphlab.boosted_trees.create(training_sframe,
                             features=new_features,
                             target='log-registered')

def fused_predict(m1, m2, test_sframe):
    """
   Fused the prediction of two separately trained models.
   The input models are trained in the log domain.
   Return the combine predictions in the original domain.
   """
    p1 = m1.predict(test_sframe).apply(lambda x: math.exp(x)-1)
    p2 = m2.predict(test_sframe).apply(lambda x: math.exp(x)-1)
    return (p1 + p2).apply(lambda x: x if x > 0 else 0)

prediction = fused_predict(m1, m2, test_sframe)



env = graphlab.deploy.environment.Local('hyperparam_search')
training = training_sframe[training_sframe['day'] <= 16]
validation = training_sframe[training_sframe['day'] > 16]
training.save('/tmp/training')
validation.save('/tmp/validation')





ntrees = 500
search_space = {
    'params': {
        'max_depth': [10, 15, 20],
        'min_child_weight': [5, 10, 20],
        'step_size': 0.05
    },
    'num_iterations': ntrees
}

def parameter_search(training_url, validation_url, default_params):
    """
   Return the optimal parameters in the given search space.
   The parameter returned has the lowest validation rmse.
   """
    job = graphlab.toolkits.model_parameter_search(env, graphlab.boosted_trees.create,
                                             train_set_path=training_url,
                                             save_path='/tmp/job_output',
                                             standard_model_params=default_params,
                                             hyper_params=search_space,
                                             test_set_path=validation_url)


    # When the job is done, the result is stored in an SFrame
    # The result contains attributes of the models in the search space
    # and the validation error in RMSE.
    result = graphlab.SFrame('/tmp/job_output').sort('rmse', ascending=True)

    # Return the parameters with the lowest validation error.
    optimal_params = result[['max_depth', 'min_child_weight']][0]
    optimal_rmse = result['rmse'][0]
    print 'Optimal parameters: %s' % str(optimal_params)
    print 'RMSE: %s' % str(optimal_rmse)
    return optimal_params




fixed_params = {'features': new_features,
                'verbose': False}

fixed_params['target'] = 'log-casual'
params_log_casual = parameter_search('/tmp/training',
                                     '/tmp/validation',
                                     fixed_params)

fixed_params['target'] = 'log-registered'
params_log_registered = parameter_search('/tmp/training',
                                         '/tmp/validation',
                                         fixed_params)




m_log_registered = graphlab.boosted_trees.create(training_sframe,
                                           features=new_features,
                                           target='log-registered',
                                           num_iterations=ntrees,
                                           params=params_log_registered,
                                           verbose=False)

m_log_casual = graphlab.boosted_trees.create(training_sframe,
                                       features=new_features,
                                       target='log-casual',
                                       num_iterations=ntrees,
                                       params=params_log_casual,
                                       verbose=False)

final_prediction = fused_predict(m_log_registered, m_log_casual, test_sframe)

make_submission(final_prediction, 'submission2.txt')




training_column_types = [str,int,int,int,int,float,float,int,float,int,int,int]
training_sframe = graphlab.SFrame.read_csv('data/train.csv', column_type_hints=training_column_types)

# load test data
test_column_types = [str,int,int,int,int,float,float,int,float]
test_sframe = graphlab.SFrame.read_csv('data/test.csv', column_type_hints=test_column_types)

