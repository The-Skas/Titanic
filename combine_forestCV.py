from data_helpers import *

# Import the random forest package

# train_test_split(train_data.values[0::,1::],train_data.values[0::,0], test_size=0.01)



arrCasualCount, _id, casual_score, df = calculateGradientModel(col_pred='casual', cols_remove=['registered'], casual=True, additional_cols_remove=[])

arrRegisteredCount, _id, registered_score, df  = calculateGradientModel(col_pred='registered', cols_remove=['casual'], casual=False, additional_cols_remove=[])

arrResult = arrRegisteredCount + arrCasualCount

write_model("data/CombineRandomForestModel.csv", arrResult, _id)
pdb.set_trace()
print "done!"
