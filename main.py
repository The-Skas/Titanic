import csv as csv
import numpy as np
import os
import pprint
import pdb

# Open up the csv file in to a Python Object
pp = pprint.PrettyPrinter(indent=4)
csv_file_object = csv.reader(open('data/train.csv', 'rb'))

# The 'next()' simply skips the first line (we skip it since its a header)
header = csv_file_object.next()

data=[]
for row in csv_file_object:
	data.append(row)

# Conver data to an array.
data = np.array(data)
# Set break point
# pdb.set_trace()
pp.pprint(data)


