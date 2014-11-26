from data_helpers import rmsle_scorer
from data_helpers import clean_data_to_numbers
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pdb

def plot_casual_registered_relation(df):
	columns = set(df.columns.values) - set(['casual', 'registered'])

	for i,column in enumerate(columns):
		fig, ax = plt.subplots()
		df.groupby(df[column]).registered.mean().plot(kind='bar',ax=ax)
		df.groupby(df[column]).casual.mean().plot(kind='bar',ax=ax, color=['r'])
		ax.legend()
df, _id = clean_data_to_numbers("data/train.csv", [])

df_tr, _id = clean_data_to_numbers("data/test.csv", [])
pdb.set_trace()
plot_casual_registered_relation(df)
# # This command will plot the Number of people who survived by sex and Pclass
# df.groupby(['Sex', 'Pclass']).Survived.sum().plot(kind='bar')

# # Survivabl: Plotting by PfClass and Sex
# death_counts= pd.crosstab([df.Pclass, df.Sex], df.Survived.astype(bool)).apply(lambda r: r, axis=1)
# death_counts.plot(kind='bar',stacked=True,color=['black','gold'], grid=False)

# death_counts.div(death_counts.sum(1).astype(float), axis = 0).plot(kind='barh', stacked=True, color=['black','gold'])

# # Select rows based on array value, can combine with apply
# df[df.Prefix.isin(['Capt.','Don.','Major.'])]





plt.show()

pdb.set_trace()
print "Done"