import csv as csv
import numpy as np
import pdb as pdb
from data_helpers import *

csv_file_object = csv.reader(open('data/LogisticRegressionModelCV.csv', 'rb')) 
csv_file_2 = csv.reader(open('data/SVCModel.csv', 'rb')) 
csv_file_3 = csv.reader(open('data/genderbasedmodel.csv', 'rb')) 

header = csv_file_object.next()  # The next() command just skips the 
csv_file_2.next()
csv_file_3.next()
                                 # first line which is a header
data=[]                          # Create a variable called 'data'.
passengers = []
for row in csv_file_object:      # Run through each row in the csv file,
	data.append(int(row[1]) + int(csv_file_2.next()[1]) +  int(csv_file_3.next()[1]))
	passengers.append(row[0])     
								 # adding each row to the data variable

pdb.set_trace()
data = np.array(data) 	         # Then convert from a list to an array
			         			 # Be aware that each item is currently
                                 # a string in this format
                                 
data = np.array(data) / 3.0
data = data.round()

write_model("data/averageModels.csv",data, passengers )