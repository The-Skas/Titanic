from data_helpers import *
import traceback, sys

casual_best_feature_scores= feature_selection_model('casual', ['registered'])
registered_best_feature_scores= feature_selection_model('registered', ['casual'])


print casual_best_feature_scores

print registered_best_feature_scores

pdb.set_trace()

print "done!"


type, value, tb = sys.exc_info()
traceback.print_exc()
pdb.post_mortem(tb)
